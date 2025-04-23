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