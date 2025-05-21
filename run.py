import torch
import threading
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
import cv2
import os
import open3d as o3d

# Tkinter의 messagebox를 사용하기 전에 루트 윈도우가 필요하다.
# 오류 발생 시 메시지를 보여주기 위해 미리 생성해 두고,
# 본격적인 UI 초기화는 하단에서 진행한다.
root = tk.Tk()
root.withdraw()

# --- 추가: RMBG 모델 임포트 ---
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
# --- 추가 끝 ---

# --------------------------------------------------
# 1) Depth 모델 및 전처리 transform 로드
# --------------------------------------------------
try:
    import depth_pro
except ImportError:
    messagebox.showerror("Import Error", "Could not import 'depth_pro'. Make sure depth_pro.py is in the same directory or accessible.")
    exit()  # Exit if the crucial module is missing

try:
    model_depth, transform = depth_pro.create_model_and_transforms()  # 변수명 'model' -> 'model_depth' 로 변경 (RMBG 모델과 구분)
    model_depth.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_depth.to(device)
    print(f"[Depth] Model loaded on device: {device}")
except Exception as e:
    messagebox.showerror("Model Loading Error", f"Failed to load depth model: {e}")  # 모델명 명시
    traceback.print_exc()  # 상세 에러 추가
    exit()

# --- 추가: RMBG 모델 로드 ---
try:
    rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    rmbg_model.to(device)  # 동일한 device 사용
    rmbg_model.eval()
    print("[RMBG] Model loaded on device:", device)
except Exception as e:
    messagebox.showerror("RMBG Model Loading Error", f"Failed to load RMBG model: {e}")
    traceback.print_exc()
    exit()

# RMBG용 Transform 정의
rmbg_image_size = (1024, 1024)
transform_rmbg = transforms.Compose([
    transforms.Resize(rmbg_image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# --- 추가 끝 ---

# 전역 변수
loaded_image = None           # depth_pro.load_rgb()로 로드한 원본 또는 RMBG 결과 (PIL.Image, (A)RGB)
f_px = None                   # 초점 길이 (픽셀 단위)
depth_result = None           # 컬러맵이 적용된 깊이 결과 (PIL.Image)
raw_depth = None              # 실제 (정규화 전) 깊이 배열 (numpy), 배경 영역은 np.nan 처리
width_height = None           # 원본 이미지 (w, h)
last_loaded_image_path = None # 로드한 이미지 파일 경로 (전체 파일 경로)

# --- 추가: 버튼 상태 관리 함수 (전역) ---
def disable_all_buttons():
    if 'load_button' in globals(): load_button.config(state=tk.DISABLED)
    if 'rmbg_button' in globals(): rmbg_button.config(state=tk.DISABLED)
    if 'submit_button' in globals(): submit_button.config(state=tk.DISABLED)
    if 'save_button' in globals(): save_button.config(state=tk.DISABLED)
    if 'view3d_button' in globals(): view3d_button.config(state=tk.DISABLED)
    if 'mesh_button' in globals(): mesh_button.config(state=tk.DISABLED)

def enable_all_buttons():
    if 'load_button' in globals(): load_button.config(state=tk.NORMAL)
    if 'rmbg_button' in globals(): rmbg_button.config(state=tk.NORMAL)
    if 'submit_button' in globals(): submit_button.config(state=tk.NORMAL)
    if 'save_button' in globals(): save_button.config(state=tk.NORMAL)
    if 'view3d_button' in globals(): view3d_button.config(state=tk.NORMAL)
    if 'mesh_button' in globals(): mesh_button.config(state=tk.NORMAL)
# --- 추가 끝 ---


# --------------------------------------------------
# 2) letterbox: 이미지 미리보기용 (변경 없음)
# --------------------------------------------------
def fit_image_to_canvas(pil_img, max_w=600, max_h=400, bg_color=(128, 128, 128)):
    if pil_img is None:
        return Image.new("RGB", (max_w, max_h), bg_color)

    w, h = pil_img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (max_w, max_h), bg_color)

    ratio = w / h
    target_ratio = max_w / max_h

    if ratio > target_ratio:
        new_w = max_w
        new_h = int(round(max_w / ratio))
    else:
        new_h = max_h
        new_w = int(round(max_h * ratio))

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    try:
        resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        resized = pil_img.resize((new_w, new_h), resample_method)
    except Exception as e:
        print(f"Error resizing image: {e}")
        traceback.print_exc()
        return Image.new("RGB", (max_w, max_h), bg_color)

    canvas = Image.new("RGB", (max_w, max_h), bg_color)
    offset_x = (max_w - new_w) // 2
    offset_y = (max_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas

# --- 추가: UI 업데이트 헬퍼 함수들 ---
def update_input_preview(img_to_show):
    """input_label을 주어진 이미지로 업데이트 (fit_image_to_canvas 사용)"""
    try:
        disp = fit_image_to_canvas(img_to_show, 600, 400)
        preview = ImageTk.PhotoImage(disp)
        input_label.config(image=preview, text="")
        input_label.image = preview  # 참조 유지!
    except Exception as e:
        print(f"Error updating input preview: {e}")
        traceback.print_exc()
        reset_input_preview(error=True)

def reset_input_preview(error=False):
    """input_label을 초기/에러 상태로 리셋"""
    text = "이미지 로드 오류" if error else "Input Image\n(Load Image)"
    blank_img = ImageTk.PhotoImage(Image.new("RGB", (600, 400), (128, 128, 128)))
    input_label.config(image=blank_img, text=text)
    input_label.image = blank_img

def update_output_preview(img_to_show, text=""):
    """output_label을 주어진 이미지로 업데이트 (fit_image_to_canvas 사용)"""
    try:
        disp = fit_image_to_canvas(img_to_show, 600, 400)
        preview = ImageTk.PhotoImage(disp)
        output_label.config(image=preview, text=text)
        output_label.image = preview  # 참조 유지!
    except Exception as e:
        print(f"Error updating output preview: {e}")
        traceback.print_exc()
        reset_output_preview(error=True)

def reset_output_preview(error=False):
    """output_label을 초기/에러 상태로 리셋"""
    text = "오류 발생" if error else "Depth Result\n(Submit)"
    blank_img = ImageTk.PhotoImage(Image.new("RGB", (600, 400), (128, 128, 128)))
    output_label.config(image=blank_img, text=text)
    output_label.image = blank_img
# --- 추가 끝 ---


# --------------------------------------------------
# 3) 이미지 로드 함수
# --------------------------------------------------
def load_image():
    global loaded_image, f_px, depth_result, raw_depth, width_height, last_loaded_image_path

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not file_path:
        return

    last_loaded_image_path = file_path

    try:
        loaded_data, _, f_px_val = depth_pro.load_rgb(file_path)
        print(f"Type returned by load_rgb: {type(loaded_data)}")

        # loaded_data를 PIL RGB 이미지로 변환/확인
        if isinstance(loaded_data, np.ndarray):
            print(f"Loaded as NumPy array. Shape: {loaded_data.shape}, Dtype: {loaded_data.dtype}")
            if loaded_data.ndim == 3 and loaded_data.shape[2] == 3:
                loaded_image = Image.fromarray(loaded_data)  # RGB로 가정
            elif loaded_data.ndim == 2:
                # 흑백인 경우
                loaded_image = Image.fromarray(cv2.cvtColor(loaded_data, cv2.COLOR_GRAY2RGB))
            else:
                raise ValueError(f"Unsupported NumPy shape: {loaded_data.shape}")

        elif isinstance(loaded_data, Image.Image):
            print("Loaded as PIL Image.")
            if loaded_data.mode != 'RGB':
                print(f"Converting PIL image from {loaded_data.mode} to RGB")
                if 'A' in loaded_data.mode:  # 알파 채널 처리 (흰색 배경 or etc)
                    # 여기서는 원본 load_rgb가 알파를 줄 가능성이 낮으니, 단순 RGB 변환 처리
                    background = Image.new('RGB', loaded_data.size, (255, 255, 255))
                    try:
                        background.paste(loaded_data, mask=loaded_data.split()[-1])
                        loaded_image = background
                    except IndexError:
                        loaded_image = loaded_data.convert('RGB')
                else:
                    loaded_image = loaded_data.convert('RGB')
            else:
                loaded_image = loaded_data
        else:
            raise TypeError(f"Unexpected type from load_rgb: {type(loaded_data)}")

        # Focal length 처리
        if f_px_val is None:
            f_px_val = max(loaded_image.size) * 1.2
            print(f"Focal length estimated: {f_px_val:.2f} pixels")
        f_px = float(f_px_val)

        width_height = loaded_image.size
        print(f"Image loaded: {file_path}, Size: {width_height}, Mode: {loaded_image.mode}, Focal: {f_px:.2f}")

        # UI 업데이트
        update_input_preview(loaded_image)
        reset_output_preview()
        depth_result = None
        raw_depth = None

    except FileNotFoundError:
        messagebox.showerror("오류", f"파일 없음: {file_path}")
    except Exception as e:
        print("--- Error during image loading ---")
        traceback.print_exc()
        messagebox.showerror("오류", f"이미지 로드 오류:\n{e}")
        loaded_image = None
        raw_depth = None
        depth_result = None
        width_height = None
        f_px = None
        last_loaded_image_path = None
        reset_input_preview(error=True)
        reset_output_preview(error=True)


# --------------------------------------------------
# (RMBG) 배경 제거
# --------------------------------------------------
def update_rmbg_ui(result_image_rgba):
    """RMBG 결과를 UI에 반영 (메인 스레드)"""
    global loaded_image, depth_result, raw_depth

    print("\n--- [DEBUG] Entered update_rmbg_ui (Main Thread) ---")
    if result_image_rgba is None:
        print("    [ERROR] result_image_rgba is None!")
        return

    try:
        # RGBA로 세팅
        loaded_image = result_image_rgba
        print(f"    Loaded_image updated => mode={loaded_image.mode}, size={loaded_image.size}")

        # 미리보기 업데이트
        update_input_preview(loaded_image)

        # 깊이 결과 리셋
        reset_output_preview()
        depth_result = None
        raw_depth = None

    except Exception as e:
        print("--- [ERROR] update_rmbg_ui ---")
        traceback.print_exc()
        messagebox.showerror("RMBG Update Error", f"RMBG 결과 적용 오류: {e}")
    finally:
        print("--- [DEBUG] Exiting update_rmbg_ui ---\n")


def remove_bg_task():
    """RMBG 실제 작업 함수 (백그라운드 스레드)"""
    current_image = None
    print("\n--- [DEBUG] Entered remove_bg_task (Background Thread) ---")
    if loaded_image is not None:
        current_image = loaded_image.copy()

    if current_image is None:
        root.after(0, lambda: messagebox.showwarning("경고", "먼저 이미지를 로드하세요."))
        print("    No image loaded, exiting remove_bg_task.")
        return

    print("[RMBG Task] 배경 제거 시작...")
    root.after(0, disable_all_buttons)

    try:
        # 1) RGB 변환
        if current_image.mode != 'RGB':
            current_image = current_image.convert("RGB")

        # 2) Inference
        input_tensor = transform_rmbg(current_image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = rmbg_model(input_tensor)[-1].sigmoid().cpu()

        pred_mask = preds[0].squeeze()
        mask_pil = transforms.ToPILImage()(pred_mask)
        mask_pil = mask_pil.resize(current_image.size, Image.Resampling.LANCZOS)

        # 3) 알파 채널 붙이기 (배경 투명)
        result_image_alpha = current_image.copy()
        result_image_alpha.putalpha(mask_pil)

        # alpha 이미지를 그대로 유지 (흰색 배경 X)
        result_image_rgba = result_image_alpha
        print("[RMBG Task] 배경 제거 완료 (alpha kept).")

        # 4) UI 업데이트 예약
        root.after(0, update_rmbg_ui, result_image_rgba)

        # 5) Depth 관련 변수 리셋
        global depth_result, raw_depth
        depth_result = None
        raw_depth = None

    except Exception as e:
        print("--- [ERROR] Error during RMBG Task ---")
        traceback.print_exc()
        root.after(0, lambda e=e: messagebox.showerror("RMBG 오류", f"배경 제거 중 오류: {e}"))
    finally:
        root.after(0, enable_all_buttons)
        print("[RMBG Task] 종료 (Finally block).\n")

def remove_bg():
    """RMBG 버튼 명령 함수"""
    threading.Thread(target=remove_bg_task, daemon=True).start()


# --------------------------------------------------
# Depth 추론 (스레드)
# --------------------------------------------------
def process_depth_task():
    global loaded_image, f_px, depth_result, raw_depth, width_height

    print("\n--- [DEBUG] Entered process_depth_task (Background Thread) ---")
    current_image_for_depth = None
    if loaded_image is not None:
        current_image_for_depth = loaded_image.copy()  # RGBA 가능
    current_f_px = f_px
    current_width_height = width_height

    if current_image_for_depth is None:
        root.after(0, lambda: messagebox.showwarning("경고", "먼저 이미지를 로드하세요."))
        print("    No image for depth, exiting process_depth_task.")
        return

    print("[Depth Task] 추론 시작...")
    root.after(0, disable_all_buttons)

    try:
        ### CHANGED/ADDED: RGBA -> RGB + alpha mask
        if current_image_for_depth.mode == 'RGBA':
            # 알파 채널 추출
            alpha_channel = current_image_for_depth.split()[-1]  # 0..255
            color_rgb = current_image_for_depth.convert("RGB")
        else:
            alpha_channel = None
            color_rgb = current_image_for_depth

        # 1) Depth 모델 추론
        with torch.no_grad():
            image_tensor = transform(color_rgb).unsqueeze(0).to(device)
            if current_f_px is not None:
                f_px_tensor = torch.tensor(current_f_px, dtype=torch.float32, device=device)
                prediction = model_depth.infer(image_tensor, f_px=f_px_tensor)
            else:
                prediction = model_depth.infer(image_tensor)

            if isinstance(prediction, dict) and "depth" in prediction:
                depth = prediction["depth"]
            elif torch.is_tensor(prediction):
                depth = prediction
            else:
                raise TypeError(f"Unexpected output type: {type(prediction)}")

        depth_np = depth.cpu().numpy().squeeze()  # HxW
        local_raw_depth = depth_np.copy()

        ### CHANGED/ADDED: alpha가 있으면 배경을 NaN 처리
        if alpha_channel is not None:
            alpha_arr = np.array(alpha_channel)  # 0..255
            # alpha=0 => 완전 투명 => 배경
            local_raw_depth[alpha_arr == 0] = np.nan

        # 2) 컬러맵용 시각화 (NaN 제외)
        valid_mask = ~np.isnan(local_raw_depth)
        if not np.any(valid_mask):
            raise ValueError("All depth values are NaN after alpha masking!")

        valid_vals = local_raw_depth[valid_mask]
        d_min, d_max = valid_vals.min(), valid_vals.max()

        if d_max - d_min < 1e-6:
            norm_depth = np.zeros_like(local_raw_depth)
        else:
            norm_depth = np.copy(local_raw_depth)
            norm_depth[valid_mask] = (norm_depth[valid_mask] - d_min) / (d_max - d_min)

        depth_uint8 = np.zeros_like(norm_depth, dtype="uint8")
        depth_uint8[valid_mask] = (norm_depth[valid_mask] * 255).astype("uint8")

        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        colored_depth_pil = Image.fromarray(colored)

        target_size = current_width_height if current_width_height else colored_depth_pil.size
        if colored_depth_pil.size != target_size:
            resample_vis = Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST
            colored_depth_img_resized = colored_depth_pil.resize(target_size, resample_vis)
        else:
            colored_depth_img_resized = colored_depth_pil

        raw_depth = local_raw_depth  # 배경영역 NaN
        depth_result = colored_depth_img_resized

        root.after(0, update_output_preview, depth_result)
        print("[Depth Task] 완료.")

    except Exception as e:
        print("--- [ERROR] Error during Depth Task ---")
        traceback.print_exc()
        root.after(0, lambda e=e: messagebox.showerror("Depth 오류", f"추론 오류: {e}"))
        raw_depth = None
        depth_result = None
        root.after(0, lambda: reset_output_preview(error=True))
    finally:
        root.after(0, enable_all_buttons)
        print("[Depth Task] 종료 (Finally block).\n")

def process_depth():
    threading.Thread(target=process_depth_task, daemon=True).start()


# --------------------------------------------------
# 5) 포인트 클라우드 생성 함수
# --------------------------------------------------
def generate_points_with_normals(raw_depth_input, color_image_input, f_px_input):
    """배경(NaN) 영역은 포인트 생성 안됨"""
    if raw_depth_input is None or color_image_input is None or f_px_input is None:
        print("Error: Missing input for PCD gen.")
        return None, None, None

    current_raw_depth = raw_depth_input.copy()
    current_color_image = color_image_input.copy()
    current_f_px = f_px_input

    # color_image_input => PIL RGB (또는 RGBA -> RGB 변환)
    if not isinstance(current_color_image, Image.Image):
        try:
            if (isinstance(current_color_image, np.ndarray) and
                current_color_image.ndim == 3 and current_color_image.shape[2] == 3):
                current_color_image = Image.fromarray(current_color_image)
            else:
                raise TypeError("Need PIL Image or compatible NumPy array.")
        except Exception as conv_err:
            print(f"Error converting to PIL: {conv_err}")
            return None, None, None

    if current_color_image.mode not in ['RGB', 'RGBA']:
        current_color_image = current_color_image.convert('RGB')

    H, W = current_raw_depth.shape[:2]
    img_W, img_H = current_color_image.size

    # 사이즈 맞추기
    if (W, H) != (img_W, img_H):
        print(f"Warning: Resizing color {current_color_image.size} -> {(W, H)} for PCD.")
        try:
            resample = Image.Resampling.NEAREST
        except AttributeError:
            resample = Image.NEAREST
        current_color_image = current_color_image.resize((W, H), resample)

    # ### CHANGED: valid_depth_mask에서 NaN 제외
    valid_depth_mask = (~np.isnan(current_raw_depth)) & (current_raw_depth > 1e-6)

    if not valid_depth_mask.any():
        print("Error: No valid depth points.")
        return None, None, None

    color_np = np.array(current_color_image)  # HxW x (3 or 4)
    cx, cy = W / 2.0, H / 2.0
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

    Z = current_raw_depth[valid_depth_mask]
    X = (u_coords[valid_depth_mask] - cx) * Z / current_f_px
    Y = (v_coords[valid_depth_mask] - cy) * Z / current_f_px
    points = np.vstack((X, Y, Z)).T

    # 색상 sampling
    if color_np.shape[2] == 4:
        # RGBA -> RGB
        color_np = color_np[..., :3]  # 앞 3채널만
    colors = color_np[valid_depth_mask] / 255.0
    if colors.shape[0] != points.shape[0]:
        print(f"Count mismatch: {colors.shape[0]} vs {points.shape[0]}")
        return None, None, None

    # 노멀 추정
    normals = None
    if points.shape[0] > 30:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print("Estimating normals...")
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=max(W, H) * 0.01, max_nn=30)
            pcd.estimate_normals(search_param=search_param)
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

            if pcd.has_normals():
                normals = np.asarray(pcd.normals)
                nan_mask = np.isnan(normals).any(axis=1)
                if np.sum(nan_mask) > 0:
                    print(f"Warning: Found NaN normals. Replacing.")
                    normals[nan_mask] = [0, 0, -1]
            else:
                print("Warning: Normal estimation failed.")
        except Exception as e:
            print(f"Could not estimate normals: {e}")
            traceback.print_exc()
    else:
        print("Warning: Not enough points for normals.")

    return points, colors, normals


# --------------------------------------------------
# 6) PLY 저장 함수 (법선 포함)
# --------------------------------------------------
def save_pointcloud_as_ply_with_normals(points, colors, normals, file_path):
    if points is None or colors is None or points.shape[0] == 0:
        print("Error: No data to save.")
        return False

    if points.shape[0] != colors.shape[0]:
        print(f"Error: Count mismatch.")
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None and normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(normals)
        print("Including normals.")
    else:
        print("Normals not included/available.")

    try:
        o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)
        print(f"PCD saved: {file_path}")
        return True
    except Exception as e:
        print(f"PLY Save Error: {e}")
        traceback.print_exc()
        return False


# --------------------------------------------------
# 7) 결과 저장 (Save)
# --------------------------------------------------
def save_result():
    if raw_depth is None:
        messagebox.showwarning("경고", "저장할 결과 없음.")
        return
    if loaded_image is None or f_px is None or width_height is None:
        messagebox.showwarning("경고", "이미지 정보 부족.")
        return

    file_path_selected = filedialog.asksaveasfilename(
        title="Save Depth Image and Point Cloud",
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All Files", "*.*")]
    )
    if not file_path_selected:
        return

    threading.Thread(target=save_result_task, args=(file_path_selected,), daemon=True).start()

def save_result_task(file_path_base):
    current_depth_result = depth_result
    current_raw_depth = raw_depth.copy() if raw_depth is not None else None
    current_loaded_image = loaded_image.copy() if loaded_image is not None else None
    current_f_px = f_px
    current_width_height = width_height

    if (current_raw_depth is None or
        current_loaded_image is None or
        current_f_px is None or
        current_width_height is None):
        print("[Save Task] Error: Data became invalid after task start.")
        return

    base, _ = os.path.splitext(file_path_base)
    png_file_path = base + "_depth_color.png"
    ply_file_path = base + "_3D.ply"
    print(f"[Save Task] Saving to:\n  PNG: {png_file_path}\n  PLY: {ply_file_path}")
    root.after(0, disable_all_buttons)

    png_saved, ply_saved = False, False
    try:
        # PNG 저장
        if current_depth_result:
            try:
                current_depth_result.save(png_file_path)
                png_saved = True
                print("PNG saved.")
            except Exception as e:
                print(f"PNG Save Error: {e}")
                traceback.print_exc()
        else:
            print("No depth result image to save.")

        # PLY 저장
        print("Preparing PLY data...")
        orig_w, orig_h = current_width_height
        depth_h, depth_w = current_raw_depth.shape

        depth_for_ply = current_raw_depth
        color_for_ply = current_loaded_image
        if (depth_w, depth_h) != (orig_w, orig_h):
            print(f"Resizing depth/color for PLY: {(orig_w, orig_h)}")
            depth_for_ply = cv2.resize(current_raw_depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            if color_for_ply.size != (orig_w, orig_h):
                try:
                    resample = Image.Resampling.NEAREST
                except:
                    resample = Image.NEAREST
                color_for_ply = color_for_ply.resize((orig_w, orig_h), resample)

        points, colors, normals = generate_points_with_normals(depth_for_ply, color_for_ply, current_f_px)
        if points is not None:
            ply_saved = save_pointcloud_as_ply_with_normals(points, colors, normals, ply_file_path)
        else:
            print("PLY Error: Point generation failed.")

    except Exception as e:
        print(f"--- Error during Save Task ---")
        traceback.print_exc()
    finally:
        root.after(0, enable_all_buttons)

        def show_status():
            if png_saved and ply_saved:
                messagebox.showinfo("저장 완료", f"저장 완료:\nPNG: {png_file_path}\nPLY: {ply_file_path}")
            elif png_saved:
                messagebox.showwarning("부분 저장", f"PNG만 저장됨:\n{png_file_path}")
            elif ply_saved:
                messagebox.showwarning("부분 저장", f"PLY만 저장됨:\n{ply_file_path}")
            else:
                messagebox.showerror("저장 실패", "모든 파일 저장 실패.")

        root.after(0, show_status)
        print("[Save Task] 종료.")


# --------------------------------------------------
# 8) Open3D 시각화 (포인트 클라우드)
# --------------------------------------------------
def view_in_open3d_task():
    current_raw_depth = raw_depth.copy() if raw_depth is not None else None
    current_loaded_image = loaded_image.copy() if loaded_image is not None else None
    current_f_px = f_px
    current_width_height = width_height

    if current_raw_depth is None:
        root.after(0, lambda: messagebox.showwarning("경고", "Depth 데이터 없음."))
        return
    if current_loaded_image is None or current_f_px is None:
        root.after(0, lambda: messagebox.showwarning("경고", "이미지/초점 정보 없음."))
        return

    if current_width_height is None:
        if current_loaded_image:
            current_width_height = current_loaded_image.size
        else:
            root.after(0, lambda: messagebox.showerror("오류", "이미지 크기 정보 없음."))
            return

    print("[View3D Task] 데이터 준비 시작...")
    root.after(0, disable_all_buttons)

    try:
        orig_w, orig_h = current_width_height
        depth_h, depth_w = current_raw_depth.shape
        depth_for_vis = current_raw_depth
        color_for_vis = current_loaded_image

        if (depth_w, depth_h) != (orig_w, orig_h):
            print(f"Resizing depth/color for Vis: {(orig_w, orig_h)}")
            depth_for_vis = cv2.resize(current_raw_depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            if color_for_vis.size != (orig_w, orig_h):
                try:
                    resample = Image.Resampling.NEAREST
                except:
                    resample = Image.NEAREST
                color_for_vis = color_for_vis.resize((orig_w, orig_h), resample)

        points, colors, normals = generate_points_with_normals(depth_for_vis, color_for_vis, current_f_px)
        if points is None or points.shape[0] == 0:
            raise ValueError("유효 포인트 생성 실패.")

        print(f"Generated {points.shape[0]} points.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None and normals.shape == points.shape:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        def run_visualizer():
            print("[View3D Task] 시각화 실행 (메인 스레드)...")
            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name=f"Open3D PCD ({orig_w}x{orig_h})", width=orig_w, height=orig_h)
                vis.add_geometry(pcd)
                opt = vis.get_render_option()
                opt.point_size = 1.0
                opt.background_color = np.asarray([0.1, 0.1, 0.1])

                # 카메라 설정
                ctr = vis.get_view_control()
                param = ctr.convert_to_pinhole_camera_parameters()
                param.intrinsic.set_intrinsics(
                    width=orig_w,
                    height=orig_h,
                    fx=current_f_px,
                    fy=current_f_px,
                    cx=orig_w / 2.0,
                    cy=orig_h / 2.0
                )
                param.extrinsic = np.eye(4)
                ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

                vis.run()
                vis.destroy_window()
                print("Visualizer closed.")
            except Exception as e:
                print(f"Visualizer Error: {e}")
                traceback.print_exc()
            finally:
                root.after(0, enable_all_buttons)

        root.after(0, run_visualizer)

    except Exception as e:
        print("--- Error preparing View3D data ---")
        traceback.print_exc()
        root.after(0, lambda e=e: messagebox.showerror("View3D 오류", f"데이터 준비 오류: {e}"))
        root.after(0, enable_all_buttons)

def view_in_open3d():
    threading.Thread(target=view_in_open3d_task, daemon=True).start()


# --------------------------------------------------
# 10) 메시 생성 (Poisson) - 텍스처 매핑
# --------------------------------------------------
def create_and_texture_mesh_task():
    global raw_depth, loaded_image, f_px, width_height, last_loaded_image_path

    current_raw_depth = raw_depth.copy() if raw_depth is not None else None
    current_loaded_image = loaded_image.copy() if loaded_image is not None else None
    current_f_px = f_px
    current_width_height = width_height
    current_last_loaded_image_path = last_loaded_image_path

    print("\n" + "="*30 + "\n[Mesh Task] 시작...")
    root.after(0, disable_all_buttons)

    mesh = None

    try:
        if current_raw_depth is None:
            raise ValueError("Depth 데이터 없음.")
        if current_loaded_image is None or current_f_px is None:
            raise ValueError("이미지/초점 정보 없음.")
        if current_width_height is None:
            if current_loaded_image:
                current_width_height = current_loaded_image.size
            else:
                raise ValueError("이미지 크기 정보 없음.")

        orig_w, orig_h = current_width_height
        print(f"  Input: {orig_w}x{orig_h}, f={current_f_px:.2f}")

        depth_for_mesh = current_raw_depth
        color_for_mesh = current_loaded_image

        depth_h, depth_w = current_raw_depth.shape
        if (depth_w, depth_h) != (orig_w, orig_h):
            print(f"  Resizing depth/color for Mesh: {(orig_w, orig_h)}")
            depth_for_mesh = cv2.resize(current_raw_depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            if color_for_mesh.size != (orig_w, orig_h):
                try:
                    resample = Image.Resampling.NEAREST
                except:
                    resample = Image.NEAREST
                color_for_mesh = color_for_mesh.resize((orig_w, orig_h), resample)

        # PCD 생성
        print("  Generating point cloud for mesh...")
        points, _, normals = generate_points_with_normals(depth_for_mesh, color_for_mesh, current_f_px)
        if points is None or points.shape[0] < 10 or normals is None:
            raise ValueError("PCD/Normal 생성 실패 또는 부족.")

        pcd_for_mesh = o3d.geometry.PointCloud()
        pcd_for_mesh.points = o3d.utility.Vector3dVector(points)
        pcd_for_mesh.normals = o3d.utility.Vector3dVector(normals)
        print(f"  PCD generated: {len(points)} points.")

        # 아웃라이어 제거
        print("  Removing outliers...")
        nb = 20
        ratio = 2.0
        cleaned_pcd, _ = pcd_for_mesh.remove_statistical_outlier(nb_neighbors=nb, std_ratio=ratio)
        print(f"  Outliers removed: {len(pcd_for_mesh.points) - len(cleaned_pcd.points)}")
        if len(cleaned_pcd.points) < 100:
            raise ValueError("Outlier 제거 후 포인트 부족.")
        pcd_for_mesh = cleaned_pcd

        # Poisson
        poisson_depth = 9
        print(f"  Poisson Reconstruction (depth={poisson_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_for_mesh, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
        )
        if not mesh.has_triangles():
            raise ValueError("Poisson 메시 생성 실패.")
        print(f"  Initial mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris.")

        # 메시 정리
        print("  Cleaning mesh...")
        densities_np = np.asarray(densities)
        if len(densities_np) > 0:
            thresh = np.quantile(densities_np, 0.01)
            mesh.remove_vertices_by_mask(densities_np < thresh)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        if not mesh.has_vertices() or not mesh.has_triangles():
            raise ValueError("메시 정리 후 데이터 없음.")
        mesh.compute_vertex_normals()
        print(f"  Cleaned mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris.")

        # 비가시 정점 제거
        print("  Removing non-visible vertices...")
        mesh_verts = np.asarray(mesh.vertices)
        visible_mask = np.zeros(len(mesh_verts), dtype=bool)
        cx, cy = orig_w / 2.0, orig_h / 2.0
        depth_tol = (depth_for_mesh.max() - depth_for_mesh.min()) * 0.05 if depth_for_mesh.max() > depth_for_mesh.min() else 0.01
        valid_z = mesh_verts[:, 2] > 1e-6
        indices_valid = np.where(valid_z)[0]
        if len(indices_valid) > 0:
            X = mesh_verts[indices_valid, 0]
            Y = mesh_verts[indices_valid, 1]
            Z = mesh_verts[indices_valid, 2]
            u = np.round(current_f_px * (X / Z) + cx).astype(int)
            v = np.round(current_f_px * (Y / Z) + cy).astype(int)
            in_bounds = (u >= 0) & (u < orig_w) & (v >= 0) & (v < orig_h)
            indices_potential = indices_valid[in_bounds]
            if len(indices_potential) > 0:
                Z_p = Z[in_bounds]
                u_p = u[in_bounds]
                v_p = v[in_bounds]
                Z_map = depth_for_mesh[v_p, u_p]
                close = np.abs(Z_p - Z_map) < depth_tol
                visible_mask[indices_potential[close]] = True

        mesh.remove_vertices_by_mask(np.logical_not(visible_mask))
        mesh.remove_unreferenced_vertices()
        print(f"  Non-visible removed. Remaining: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris.")
        if not mesh.has_vertices() or not mesh.has_triangles():
            raise ValueError("비가시 제거 후 데이터 없음.")
        mesh.compute_vertex_normals()

        # UV 텍스처
        print("  Applying UVs and texture...")
        mesh_verts_final = np.asarray(mesh.vertices)
        mesh_tris_final = np.asarray(mesh.triangles)

        if len(mesh_verts_final) == 0:
            raise ValueError("No vertices for UV mapping.")

        border = 2
        tex_w = orig_w + 2 * border
        tex_h = orig_h + 2 * border
        tex_img = Image.new('RGB', (tex_w, tex_h), (128, 128, 128))
        tex_img.paste(color_for_mesh.convert('RGB'), (border, border))

        Xf = mesh_verts_final[:, 0]
        Yf = mesh_verts_final[:, 1]
        Zf = np.maximum(mesh_verts_final[:, 2], 1e-6)
        uf = current_f_px * (Xf / Zf) + cx
        vf = current_f_px * (Yf / Zf) + cy

        uv_u = np.clip((uf + border) / tex_w, 0.0, 1.0)
        uv_v = np.clip((vf + border) / tex_h, 0.0, 1.0)
        vert_uvs = np.stack([uv_u, uv_v], axis=-1)

        if len(mesh_tris_final) > 0:
            if mesh_tris_final.max() >= len(vert_uvs):
                raise IndexError("Triangle index out of UV bounds.")

            tri_uvs_flat = vert_uvs[mesh_tris_final.flatten()]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs_flat)
            mesh.textures = [o3d.geometry.Image(np.array(tex_img))]
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh_tris_final), dtype=np.int32))
            if mesh.has_vertex_colors():
                mesh.vertex_colors.clear()
            print("  UVs and texture applied.")
        else:
            print("  Warning: No triangles for UVs.")

        def run_mesh_vis():
            print("[Mesh Task] 시각화 실행 (메인 스레드)...")
            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="Textured Mesh", width=orig_w, height=orig_h)
                vis.add_geometry(mesh)
                opt = vis.get_render_option()
                opt.background_color = np.asarray([0.1, 0.1, 0.1])
                opt.mesh_show_back_face = False

                ctr = vis.get_view_control()
                param = ctr.convert_to_pinhole_camera_parameters()
                param.intrinsic.set_intrinsics(
                    width=orig_w,
                    height=orig_h,
                    fx=current_f_px,
                    fy=current_f_px,
                    cx=orig_w / 2.0,
                    cy=orig_h / 2.0
                )
                param.extrinsic = np.eye(4)
                ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

                vis.run()
                vis.destroy_window()
                print("Mesh visualizer closed.")
                root.after(0, lambda: messagebox.showinfo("성공", "메시 생성 및 시각화 완료."))

                # 자동 저장
                if current_last_loaded_image_path:
                    base, _ = os.path.splitext(current_last_loaded_image_path)
                    obj_f = base + "_mesh.obj"
                    print(f"  Saving mesh automatically: {obj_f}")
                    try:
                        success = o3d.io.write_triangle_mesh(obj_f, mesh, write_triangle_uvs=True)
                        if success:
                            root.after(0, lambda: messagebox.showinfo("저장 완료", f"메시 저장됨:\n{obj_f}"))
                        else:
                            raise IOError("write_triangle_mesh failed")
                    except Exception as save_e:
                        print(f"Mesh Save Error: {save_e}")
                        traceback.print_exc()
                else:
                    print("  Skipping auto-save (no original path).")

            except Exception as e:
                print(f"Mesh Visualizer Error: {e}")
                traceback.print_exc()
            finally:
                root.after(0, enable_all_buttons)

        root.after(0, run_mesh_vis)

    except Exception as e:
        print(f"--- Error during Mesh Task ---")
        traceback.print_exc()
        root.after(0, lambda e=e: messagebox.showerror("메시 오류", f"오류 발생:\n{e}"))
        root.after(0, enable_all_buttons)

def process_mesh_creation():
    threading.Thread(target=create_and_texture_mesh_task, daemon=True).start()


# --------------------------------------------------
# 11) Tkinter UI 설정
# --------------------------------------------------
root.deiconify()
root.title("ML Depth Pro - Simple")
root.geometry("1300x750")
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0)
root.grid_columnconfigure(0, weight=1)

top_frame = tk.Frame(root, bg="#cccccc")
top_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
top_frame.grid_rowconfigure(0, weight=1)
top_frame.grid_columnconfigure(0, weight=1)
top_frame.grid_columnconfigure(1, weight=1)

bottom_frame = tk.Frame(root, bg="#eeeeee", height=60)
bottom_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
bottom_frame.grid_columnconfigure(0, weight=1)  # 왼쪽 공간
bottom_frame.grid_columnconfigure(7, weight=1)  # 오른쪽 공간 (버튼 6개 + 1)

label_bg_color = "#e0e0e0"
input_label = tk.Label(top_frame, text="Input Image\n(Load Image)", bg=label_bg_color, width=80, height=25)
input_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
output_label = tk.Label(top_frame, text="Depth Result\n(Submit)", bg=label_bg_color, width=80, height=25)
output_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

button_width = 15
button_pady = 10
button_padx = 5

load_button = tk.Button(bottom_frame, text="Load Image", command=load_image, width=button_width)
load_button.grid(row=0, column=1, padx=button_padx, pady=button_pady)

# --- RMBG ---
rmbg_button = tk.Button(bottom_frame, text="Submit(RMBG)", command=remove_bg, width=button_width)
rmbg_button.grid(row=0, column=2, padx=button_padx, pady=button_pady)

submit_button = tk.Button(bottom_frame, text="Submit(Depth)", command=process_depth, width=button_width)
submit_button.grid(row=0, column=3, padx=button_padx, pady=button_pady)

save_button = tk.Button(bottom_frame, text="Save Results", command=save_result, width=button_width)
save_button.grid(row=0, column=4, padx=button_padx, pady=button_pady)

view3d_button = tk.Button(bottom_frame, text="View 3D (Points)", command=view_in_open3d, width=button_width)
view3d_button.grid(row=0, column=5, padx=button_padx, pady=button_pady)

mesh_button = tk.Button(bottom_frame, text="Create Mesh", command=process_mesh_creation, width=button_width)
mesh_button.grid(row=0, column=6, padx=button_padx, pady=button_pady)

reset_input_preview()
reset_output_preview()

root.mainloop()
