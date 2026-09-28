"""Fire & Smoke Detection — Flask app using rabahdev/fire-smoke-yolov8n."""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

HF_REPO_ID = "rabahdev/fire-smoke-yolov8n"
HF_WEIGHTS = "best.pt"
MODEL_PATH = MODELS_DIR / "fire-smoke-yolov8n.pt"

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
CONF_THRESHOLD = 0.25
ALERT_CLASSES = {"fire", "smoke"}
BOX_COLORS = {
    "fire": (40, 40, 220),
    "smoke": (200, 160, 80),
}

# Speed settings (CPU-friendly)
INFER_IMGSZ = 320
MAX_OUTPUT_WIDTH = 960
DETECT_FPS = 6  # run model ~6 times per second; reuse boxes between
MAX_PROCESS_SECONDS = 90  # only process first N seconds of long videos

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

_model: YOLO | None = None


def ensure_model_weights() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"Downloading {HF_REPO_ID}/{HF_WEIGHTS} ...")
    ckpt = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS)
    shutil.copy2(ckpt, MODEL_PATH)
    return MODEL_PATH


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(ensure_model_weights()))
    return _model


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def resize_frame(frame, max_width: int = MAX_OUTPUT_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    new_size = (max_width, int(round(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), scale


def draw_label(frame, name: str, score: float, x1: int, y1: int, x2: int, y2: int) -> None:
    color = BOX_COLORS.get(name.lower(), (0, 255, 255))
    label = f"{name.upper()}: {round(score * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    pad = 5
    h, w = frame.shape[:2]

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    box_h = th + pad * 2 + baseline
    above = y1 - box_h
    if above >= 0:
        ly1, ly2 = above, y1
    else:
        ly1 = max(0, min(y1, h - box_h))
        ly2 = ly1 + box_h

    lx1 = max(0, min(x1, w - tw - pad * 2))
    lx2 = min(w, lx1 + tw + pad * 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(
        frame,
        label,
        (lx1 + pad, ly2 - pad - baseline),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def process_video(input_path: Path, output_path: Path, conf: float = CONF_THRESHOLD) -> dict:
    """Detect fire/smoke with frame skipping + smaller imgsz for CPU speed."""
    model = get_model()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(src_fps / DETECT_FPS)))
    max_frames = int(src_fps * MAX_PROCESS_SECONDS) if src_fps > 0 else 0
    out_fps = max(src_fps / stride, 1.0)

    writer = None
    class_counts: dict[str, int] = {}
    frame_hits: list[dict] = []
    fire_detected = False
    smoke_detected = False
    frame_idx = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        # Sample frames for speed; keep video duration correct via out_fps.
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        frame, _ = resize_frame(frame)
        if writer is None:
            out_h, out_w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                out_fps,
                (out_w, out_h),
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Could not create output video")

        result = model.predict(frame, conf=conf, imgsz=INFER_IMGSZ, verbose=False)[0]
        frame_classes: set[str] = set()

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                name = result.names.get(int(box.cls[0]), str(int(box.cls[0])))
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                draw_label(frame, name, score, x1, y1, x2, y2)

                class_counts[name] = class_counts.get(name, 0) + 1
                lower = name.lower()
                frame_classes.add(lower)
                if lower == "fire":
                    fire_detected = True
                elif lower == "smoke":
                    smoke_detected = True

        if frame_classes & ALERT_CLASSES:
            frame_hits.append(
                {
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / src_fps, 2),
                    "classes": sorted(frame_classes & ALERT_CLASSES),
                }
            )

        writer.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if written == 0:
        raise RuntimeError("No frames could be read from the video")

    return {
        "frames_processed": written,
        "detect_stride": stride,
        "total_frames": total_frames,
        "fps": round(src_fps, 2),
        "class_counts": class_counts,
        "fire_detected": fire_detected,
        "smoke_detected": smoke_detected,
        "alert": fire_detected or smoke_detected,
        "alert_events": frame_hits[:50],
        "alert_event_count": len(frame_hits),
        "truncated": bool(max_frames and total_frames > max_frames),
    }


def try_reencode_h264(src: Path, dst: Path) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(dst),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        return dst.exists() and dst.stat().st_size > 0
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Unsupported format. Allowed: {allowed}"}), 400

    conf = max(0.05, min(0.95, float(request.form.get("conf", CONF_THRESHOLD))))

    job_id = uuid.uuid4().hex[:12]
    ext = Path(secure_filename(file.filename)).suffix.lower() or ".mp4"
    input_path = UPLOAD_DIR / f"{job_id}_input{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_detected.mp4"
    file.save(input_path)

    try:
        started = time.time()
        summary = process_video(input_path, output_path, conf=conf)
        elapsed = round(time.time() - started, 2)
    except (OSError, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 500

    web_path = OUTPUT_DIR / f"{job_id}_web.mp4"
    video_file = (
        web_path.name if try_reencode_h264(output_path, web_path) else output_path.name
    )

    return jsonify(
        {
            "job_id": job_id,
            "elapsed_sec": elapsed,
            "video_url": url_for("serve_output", filename=video_file),
            "summary": summary,
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": HF_REPO_ID,
            "model_exists": MODEL_PATH.exists(),
            "model_path": str(MODEL_PATH),
            "classes": (
                list(get_model().names.values()) if _model is not None else ["smoke", "fire"]
            ),
        }
    )


if __name__ == "__main__":
    print(f"Loading model {HF_REPO_ID} ...")
    get_model()
    print(f"Model ready: {MODEL_PATH.name}. Starting server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
