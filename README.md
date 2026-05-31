# 🎭 Live Emotion Detector

Real-time facial emotion recognition from your webcam — no model training needed. Uses a pretrained Vision Transformer (ViT) fine-tuned on FER2013 + AffectNet, downloaded automatically on first run.

## Features

- Detects and labels 7 emotions: **angry, disgust, fear, happy, neutral, sad, surprise**
- Live probability bar charts per face
- FPS counter and pause/screenshot controls
- Runs on CPU or GPU (auto-detected)

## Requirements

```bash
pip install transformers torch torchvision opencv-python pillow
```

## Usage

```bash
python live_emotion_detector.py
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--camera N` | `0` | Camera device index |
| `--conf 0.0–1.0` | `0.55` | Face detection confidence threshold |
| `--width` / `--height` | `1280×720` | Capture resolution |
| `--no_bars` | — | Disable probability bar charts |

## Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `P` | Pause / unpause |
| `S` | Save screenshot |
| `B` | Toggle probability bars |
| `H` | Show / hide controls menu |

## Models

- **Emotion:** [`mo-thecreator/vit-Facial-Expression-Recognition`](https://huggingface.co/mo-thecreator/vit-Facial-Expression-Recognition) (~330 MB, cached after first run)
- **Face detector:** OpenCV ResNet SSD (downloaded automatically to `./face_detector_weights/`)

## Notes

- First launch downloads ~330 MB of model weights — subsequent runs use the local cache.
- GPU inference (CUDA) is used automatically if available.
- Emotion inference runs every 3rd frame to keep the UI responsive.
