import cv2
import base64
import socketio
import time
import threading
from ultralytics import YOLO

print("Загрузка YOLOv8...")
model = YOLO("best.pt")
print("Модель загружена успешно!")

NODE_URL = "https://localhost:3000"
CONFIDENCE_THRESHOLD = 0.5
ANALYSIS_INTERVAL_SECONDS = 300

sio = socketio.Client(reconnection=True, ssl_verify=False)

@sio.event
def connect():
    print("Подключено к Node.js серверу!")

@sio.event
def disconnect():
    print("Отключено от сервера!")

# ---------------------------------------------------------
# НАСТРОЙКИ ДЛЯ 2 КАМЕР
# Обрати внимание на конец ссылки: Channels/101 и Channels/102
# ---------------------------------------------------------
CAMERAS = [
    # Первая камера (старый пароль, IP .237)
    {"id": 1, "url": "rtsp://admin:112233**gsc@172.16.11.237:554/Streaming/Channels/102"},
    
    # Вторая камера (НОВЫЙ пароль, IP .236)
    {"id": 2, "url": "rtsp://admin:88888888a@172.16.11.236:554/Streaming/Channels/101"}
]

# Статус ИИ для 2 камер
ai_enabled = {1: True, 2: True}

@sio.on("set_ai")
def set_ai(data):
    cam_id = int(data.get("cameraId"))
    ai_enabled[cam_id] = bool(data.get("enabled"))
    print(f"Статус ИИ для камеры {cam_id} изменен на: {ai_enabled[cam_id]}")

def connect_to_server():
    while True:
        try:
            sio.connect(NODE_URL, wait_timeout=10)
            return
        except Exception as e:
            print(f"Ошибка подключения к серверу: {e}. Повтор через 3 сек.")
            time.sleep(3)

connect_to_server()

def pick_best_detection(results):
    best = None
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls = int(box.cls[0])
            class_name = str(model.names[cls]).lower()
            if best is None or conf > best["confidence"]:
                best = {
                    "class_name": class_name,
                    "confidence": conf,
                    "box": tuple(map(int, box.xyxy[0]))
                }
    return best

def process_camera(cam_id, rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Запуск потока камеры {cam_id}...")

    prev_time = time.time()
    last_analysis_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Камера {cam_id} потеряла связь. Переподключение...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame = cv2.resize(frame, (640, 360))

        current_time = time.time()
        fps = int(1 / max(current_time - prev_time, 0.001))
        prev_time = current_time

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        
        if success:
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            sio.emit("camera_frame", {
                "type": "live",
                "cameraId": cam_id,
                "image": frame_base64,
                "fps": fps,
                "aiEnabled": ai_enabled.get(cam_id, True)
            })

        should_analyze = (
            ai_enabled.get(cam_id, True)
            and current_time - last_analysis_time >= ANALYSIS_INTERVAL_SECONDS
        )

        if should_analyze:
            last_analysis_time = current_time
            analysis_frame = frame.copy()
            trash_status = "unknown"
            confidence = 0.0

            results = model(analysis_frame, verbose=False, imgsz=640)
            detection = pick_best_detection(results)

            if detection:
                trash_status = detection["class_name"]
                confidence = detection["confidence"]
                x1, y1, x2, y2 = detection["box"]
                color = (0, 0, 255) if trash_status == "full" else (0, 255, 0)
                cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(analysis_frame, f"{trash_status} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            success, buffer = cv2.imencode(".jpg", analysis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if success:
                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                sio.emit("camera_frame", {
                    "type": "analysis",
                    "cameraId": cam_id,
                    "image": frame_base64,
                    "trashStatus": trash_status,
                    "confidence": confidence,
                    "fps": fps,
                    "aiEnabled": ai_enabled.get(cam_id, True),
                    "analysisIntervalSeconds": ANALYSIS_INTERVAL_SECONDS,
                    "analyzedAt": int(current_time * 1000)
                })
        
        time.sleep(0.01)

threads = []
for cam in CAMERAS:
    t = threading.Thread(target=process_camera, args=(cam["id"], cam["url"]))
    t.daemon = True
    t.start()
    threads.append(t)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nОстановка всех камер и отключение...")
    sio.disconnect()
    
#88888888a    
