# Fire & Smoke Detection

Web application that detects **fire** and **smoke** in uploaded videos using a YOLOv8 model, draws bounding boxes with class names and confidence scores, plays the annotated video, and shows an on-screen alert badge when either class is found.

## Features

- Upload video (MP4, MOV, AVI, MKV, WEBM)
- YOLOv8 inference with [rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n)
- Bounding boxes labeled as `FIRE: 94%` / `SMOKE: 87%`
- On-screen alert badge: `FIRE DETECTED` / `SMOKE DETECTED` / both
- Adjustable confidence threshold
- CPU-friendly processing (frame sampling + smaller inference size)

## Project structure

```
smoke-detection/
├── app.py                 # Flask backend + detection pipeline
├── templates/
│   └── index.html         # Upload UI and results view
├── models/
│   └── fire-smoke-yolov8n.pt   # Model weights (auto-download if missing)
├── uploads/               # Temporary uploaded videos
├── outputs/               # Annotated result videos
├── demo.mp4               # Sample video for testing
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- `ffmpeg` (recommended, for browser-playable H.264 output)
- CPU is enough; GPU optional

Install ffmpeg on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

## Setup

```bash
# 1. Go to the project folder
cd smoke-detection

# 2. (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python packages
pip install -r requirements.txt

# 4. Run the app
python3 app.py
```

Open in the browser:

```
http://127.0.0.1:5000
```

On first run (if `models/fire-smoke-yolov8n.pt` is missing), weights are downloaded from Hugging Face automatically.

## How to use

1. Click **Choose video** or drop a file into the drop zone  
2. Set the confidence slider (lower = more detections, higher = stricter)  
3. Click **Check video**  
4. Wait for processing (short clips are typically under a minute on CPU)  
5. Watch the annotated video; a red badge appears if fire or smoke is detected  

You can test quickly with the included `demo.mp4`.

## Implementation

### Model

| Item | Detail |
|------|--------|
| Source | [rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) |
| Base | YOLOv8n fine-tuned on D-Fire |
| Classes | `0` = smoke, `1` = fire |
| Local path | `models/fire-smoke-yolov8n.pt` |

### Pipeline (`app.py`)

1. **Upload** — video saved under `uploads/`  
2. **Frame sampling** — detect at ~6 FPS instead of every frame (faster on CPU)  
3. **Resize** — output width capped at 960px; YOLO `imgsz=320`  
4. **Inference** — Ultralytics `YOLO.predict()` per sampled frame  
5. **Draw** — OpenCV rectangles + labels (`CLASS: xx%`)  
6. **Encode** — write annotated MP4; re-encode with ffmpeg to H.264 when available  
7. **Response** — JSON with video URL, timings, class counts, and alert timeline  
8. **UI** — plays the result and shows the alert badge when `fire` / `smoke` is present  

Long videos are limited to the first **90 seconds** for responsive demos.

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/detect` | POST | `multipart/form-data`: `video`, optional `conf` |
| `/outputs/<file>` | GET | Serve annotated video |
| `/health` | GET | Model / app status |

Example detect request:

```bash
curl -X POST -F "video=@demo.mp4" -F "conf=0.25" http://127.0.0.1:5000/api/detect
```

### Tech stack

- **Backend:** Flask  
- **Vision:** Ultralytics YOLOv8, OpenCV  
- **Model hosting:** Hugging Face Hub  
- **Frontend:** HTML / CSS / JavaScript (no build step)  

## Configuration

Tunable constants at the top of `app.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `CONF_THRESHOLD` | `0.25` | Default confidence |
| `INFER_IMGSZ` | `320` | YOLO input size |
| `MAX_OUTPUT_WIDTH` | `960` | Max output frame width |
| `DETECT_FPS` | `6` | Detection rate (sampled) |
| `MAX_PROCESS_SECONDS` | `90` | Max video length to process |

## Troubleshooting

**Slow processing**  
Normal on CPU for long/high-res clips. Keep the tab open until the request finishes. Prefer shorter videos for demos.

**Video plays but no picture / won’t play**  
Install `ffmpeg` so results are re-encoded to H.264.

**Model download fails**  
Check internet access to Hugging Face, or place `best.pt` manually at:

```
models/fire-smoke-yolov8n.pt
```

**Port already in use**

```bash
fuser -k 5000/tcp
python3 app.py
```

## License notes

- App code in this repo: use as needed for your project/assignment.  
- YOLOv8 / Ultralytics weights inherit [AGPL-3.0](https://github.com/ultralytics/ultralytics) terms from Ultralytics.  
- Model card: [rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n)  
