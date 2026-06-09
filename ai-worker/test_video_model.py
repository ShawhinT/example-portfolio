import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_COLORS = {
    "full": (0, 0, 255),
    "half": (0, 255, 255),
    "empty": (0, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test trash-bin YOLO model on a local video file.")
    parser.add_argument("video", nargs="?", default="trash_test.mp4", help="Path to video file. Default: trash_test.mp4")
    parser.add_argument("--model", default="best.pt", help="Path to YOLO model. Default: best.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold. Default: 0.5")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size. Default: 640")
    parser.add_argument("--save", default="", help="Optional output video path, for example result.mp4")
    return parser.parse_args()


def draw_detections(frame, result, names, conf_threshold):
    best_status = "unknown"
    best_confidence = 0.0

    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < conf_threshold:
            continue

        class_id = int(box.cls[0])
        class_name = str(names[class_id]).lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if confidence > best_confidence:
            best_status = class_name
            best_confidence = confidence

    return best_status, best_confidence


def main():
    args = parse_args()
    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    print("Model classes:", model.names)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, source_fps, (width, height))
        print(f"Saving result to: {args.save}")

    paused = False
    prev_time = time.time()

    print("Controls: q = quit, space = pause/resume")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(frame, verbose=False, imgsz=args.imgsz)
            status, confidence = draw_detections(frame, results[0], model.names, args.conf)

            now = time.time()
            fps = 1 / max(now - prev_time, 0.001)
            prev_time = now

            header_color = CLASS_COLORS.get(status, (255, 255, 255))
            cv2.rectangle(frame, (0, 0), (width, 52), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Status: {status} | Confidence: {confidence:.2f} | FPS: {fps:.1f}",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                header_color,
                2,
            )

            if writer:
                writer.write(frame)

        cv2.imshow("Trash model video test", frame)
        key = cv2.waitKey(1 if not paused else 50) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
