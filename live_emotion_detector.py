"""
live_emotion_detector.py
========================
Real-time emotion detection — NO training required.

Uses a pretrained ViT model (trained on FER2013 + AffectNet) downloaded
automatically from HuggingFace on first run (~330 MB, cached after that).

Requirements:
    pip install transformers torch torchvision opencv-python pillow

Usage:
    python live_emotion_detector.py

Controls:
    Q / ESC  – quit
    S        – save screenshot
    P        – pause / unpause
    B        – toggle probability bar charts
    H        – show / hide controls menu
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# EMOTION CONFIG
# ---------------------------------------------------------------------------
EMOTION_COLORS_BGR = {
    "angry":    (0,   0,  255),
    "disgust":  (30, 140,  30),
    "fear":     (220,  0, 150),
    "happy":    (0,  255, 255),
    "neutral":  (200, 200, 200),
    "sad":      (255,  80,   0),
    "surprise": (0,  120, 255),
}
DEFAULT_COLOR = (0, 0, 255)

HF_MODEL = "mo-thecreator/vit-Facial-Expression-Recognition"

FACE_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
FACE_WEIGHTS_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20180205_fp16/"
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Controls menu entries: (key label, description)
CONTROLS = [
    ("Q / ESC", "Quit"),
    ("P",       "Pause / unpause"),
    ("S",       "Save screenshot"),
    ("B",       "Toggle prob. bars"),
    ("H",       "Show / hide this menu"),
]


# ---------------------------------------------------------------------------
# SETUP HELPERS
# ---------------------------------------------------------------------------
def download_if_missing(url: str, dest: Path):
    if not dest.exists():
        print(f"[INFO] Downloading {dest.name} …")
        urllib.request.urlretrieve(url, dest)
        print(f"[INFO] Saved → {dest}")


def load_face_detector(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    proto   = model_dir / "deploy.prototxt"
    weights = model_dir / "res10_300x300_ssd.caffemodel"
    download_if_missing(FACE_PROTOTXT_URL, proto)
    download_if_missing(FACE_WEIGHTS_URL,  weights)
    return cv2.dnn.readNetFromCaffe(str(proto), str(weights))


def load_emotion_pipeline():
    from transformers import pipeline as hf_pipeline
    print(f"[INFO] Loading emotion model: {HF_MODEL}")
    print("[INFO] (Downloads ~330 MB on first run, then cached)")
    pipe = hf_pipeline(
        "image-classification",
        model=HF_MODEL,
        device=0 if torch.cuda.is_available() else -1,
    )
    print("[INFO] Emotion model ready.")
    return pipe


# ---------------------------------------------------------------------------
# FACE DETECTION
# ---------------------------------------------------------------------------
def detect_faces(net, frame: np.ndarray, conf_thresh: float = 0.55):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_thresh:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


# ---------------------------------------------------------------------------
# EMOTION PREDICTION
# ---------------------------------------------------------------------------
def predict_emotions(pipe, face_crops_rgb: list):
    pil_images = [Image.fromarray(c) for c in face_crops_rgb]
    results = pipe(pil_images, top_k=None)
    outputs = []
    for res in results:
        probs = {r["label"]: r["score"] for r in res}
        best  = max(res, key=lambda r: r["score"])
        outputs.append((best["label"], best["score"], probs))
    return outputs


# ---------------------------------------------------------------------------
# DRAWING — face annotations
# ---------------------------------------------------------------------------
def draw_result(frame, x1, y1, x2, y2, label, confidence, probs, show_bars):
    color = EMOTION_COLORS_BGR.get(label, DEFAULT_COLOR)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    caption = f"{label.capitalize()}  {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(caption, font, 0.65, 2)
    badge_top = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, badge_top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, caption, (x1 + 4, y1 - 5),
                font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    if not show_bars or not probs:
        return

    bar_x     = x1
    bar_y     = y2 + 6
    bar_w_max = max(80, x2 - x1)
    bar_h     = 11
    gap       = 2

    for idx, (em, prob) in enumerate(sorted(probs.items(), key=lambda kv: kv[1], reverse=True)):
        c   = EMOTION_COLORS_BGR.get(em, DEFAULT_COLOR)
        bw  = int(prob * bar_w_max)
        bx1 = bar_x
        by1 = bar_y + idx * (bar_h + gap)
        bx2 = bx1 + bw
        by2 = by1 + bar_h
        cv2.rectangle(frame, (bx1, by1), (bx1 + bar_w_max, by2), (40, 40, 40), -1)
        if bw > 0:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), c, -1)
        if em == label:
            cv2.rectangle(frame, (bx1, by1), (bx1 + bar_w_max, by2), color, 1)
        cv2.putText(frame, em[:3].upper(),
                    (bx1 + bar_w_max + 3, by2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (210, 210, 210), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# DRAWING — HUD (top-left status bar)
# ---------------------------------------------------------------------------
def draw_hud(frame, fps, num_faces, paused):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (240, 62), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    status = "PAUSED" if paused else "LIVE"
    cv2.putText(frame, f"FPS: {fps:5.1f}   {status}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {num_faces}   H = controls", (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# DRAWING — Controls menu overlay (bottom-right)
# ---------------------------------------------------------------------------
def draw_controls_menu(frame):
    """
    Renders a semi-transparent controls panel in the bottom-right corner.
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 1
    pad_x      = 14
    pad_y      = 10
    row_h      = 22
    key_col_w  = 76   # fixed width for the key column

    # Measure widest description to size the panel
    max_desc_w = max(
        cv2.getTextSize(desc, font, font_scale, thickness)[0][0]
        for _, desc in CONTROLS
    )
    panel_w = pad_x * 2 + key_col_w + 10 + max_desc_w
    panel_h = pad_y * 2 + len(CONTROLS) * row_h + 20  # +20 for header

    h, w = frame.shape[:2]
    margin   = 14
    px1      = w - panel_w - margin
    py1      = h - panel_h - margin
    px2      = w - margin
    py2      = h - margin

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Panel border
    cv2.rectangle(frame, (px1, py1), (px2, py2), (80, 80, 80), 1)

    # Header
    header_y = py1 + pad_y + 14
    cv2.putText(frame, "Controls", (px1 + pad_x, header_y),
                font, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

    # Divider
    div_y = header_y + 6
    cv2.line(frame, (px1 + pad_x, div_y), (px2 - pad_x, div_y), (60, 60, 60), 1)

    # Rows
    for i, (key, desc) in enumerate(CONTROLS):
        row_y = div_y + pad_y + i * row_h + 14

        # Key badge background
        (kw, kh), _ = cv2.getTextSize(key, font, font_scale, thickness)
        kx1 = px1 + pad_x
        ky1 = row_y - kh - 3
        kx2 = kx1 + kw + 8
        ky2 = row_y + 3
        cv2.rectangle(frame, (kx1, ky1), (kx2, ky2), (50, 50, 50), -1)
        cv2.rectangle(frame, (kx1, ky1), (kx2, ky2), (90, 90, 90), 1)

        # Key text
        cv2.putText(frame, key, (kx1 + 4, row_y),
                    font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)

        # Description
        desc_x = px1 + pad_x + key_col_w + 10
        cv2.putText(frame, desc, (desc_x, row_y),
                    font, font_scale, (160, 160, 160), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live emotion detector (pretrained ViT)")
    parser.add_argument("--camera",      type=int,   default=0)
    parser.add_argument("--conf",        type=float, default=0.55)
    parser.add_argument("--width",       type=int,   default=1280)
    parser.add_argument("--height",      type=int,   default=720)
    parser.add_argument("--no_bars",     action="store_true")
    parser.add_argument("--face_models", default="./face_detector_weights")
    args = parser.parse_args()

    face_net     = load_face_detector(Path(args.face_models))
    emotion_pipe = load_emotion_pipeline()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    print("[INFO] Camera open.  H = controls menu")

    show_bars      = not args.no_bars
    show_controls  = True    # visible by default on launch
    paused         = False
    fps            = 0.0
    frame_count    = 0
    t_fps          = time.time()
    last_frame     = None
    screenshot_idx = 0
    INFER_EVERY    = 3
    frame_idx      = 0
    cached_results = []

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('b'):
            show_bars = not show_bars
        elif key == ord('h'):
            show_controls = not show_controls
        elif key == ord('s') and last_frame is not None:
            fname = f"screenshot_{screenshot_idx:04d}.jpg"
            cv2.imwrite(fname, last_frame)
            print(f"[INFO] Saved {fname}")
            screenshot_idx += 1

        if paused and last_frame is not None:
            cv2.imshow("Emotion Detector", last_frame)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        frame_idx   += 1
        elapsed = time.time() - t_fps
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            t_fps = time.time()

        if frame_idx % INFER_EVERY == 1:
            boxes = detect_faces(face_net, frame, args.conf)
            crops = [
                cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                for (x1, y1, x2, y2) in boxes
                if frame[y1:y2, x1:x2].size > 0
            ]
            if crops:
                try:
                    preds = predict_emotions(emotion_pipe, crops)
                    cached_results = [(boxes[i], preds[i]) for i in range(len(crops))]
                except Exception as e:
                    print(f"[WARN] {e}")
                    cached_results = []
            else:
                cached_results = []

        for (x1, y1, x2, y2), (label, conf, probs) in cached_results:
            draw_result(frame, x1, y1, x2, y2, label, conf, probs, show_bars)

        draw_hud(frame, fps, len(cached_results), paused)

        if show_controls:
            draw_controls_menu(frame)

        last_frame = frame.copy()
        cv2.imshow("Emotion Detector", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited.")


if __name__ == "__main__":
    main()