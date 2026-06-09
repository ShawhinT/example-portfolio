import cv2
import base64
import socketio
import time
from ultralytics import YOLO
import threading
import os
import urllib.parse 

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

SERVER_URLS = [
    os.environ.get("AI_SERVER_URL"),
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://172.16.11.190:3000",
]
SERVER_URLS = list(dict.fromkeys(url for url in SERVER_URLS if url))
FRAME_SIZE = (640, 360)
ANALYSIS_INTERVAL_SECONDS = 300 

CONF_THRESHOLD = 0.10 
IOU_THRESHOLD = 0.60

sio = socketio.Client(ssl_verify=False)

print("Загрузка YOLOv8...")
model = YOLO("best6.pt") 

ai_status = {}
active_cameras = {}

def emit_camera_status(cam_id, online, status, error=""):
    sio.emit("camera_status", {
        "cameraId": cam_id,
        "online": online,
        "status": status,
        "error": error,
        "updatedAt": int(time.time() * 1000)
    })

def build_rtsp_url(data):
    direct_url = str(data.get("url") or data.get("rtspUrl") or "").strip()
    ip_or_url = str(data.get("ip") or "").strip()

    if direct_url.startswith("rtsp://"):
        return direct_url
    if ip_or_url.startswith("rtsp://"):
        return ip_or_url

    usr = str(data.get("username", "admin")).strip()
    raw_pwd = str(data.get("password", "")).strip()
    ip = ip_or_url
    port = str(data.get("port", "554")).strip() or "554"
    channel = str(data.get("channel", "101")).strip().strip("/")

    if not ip:
        raise ValueError("IP камеры пустой")

    safe_usr = urllib.parse.quote(usr, safe="")
    safe_pwd = urllib.parse.quote(raw_pwd, safe="*")
    auth_part = safe_usr
    if safe_pwd:
        auth_part += f":{safe_pwd}"

    if channel.lower().startswith("streaming/channels/"):
        path = channel
    elif "/" in channel:
        path = channel
    else:
        path = f"Streaming/Channels/{channel}"

    return f"rtsp://{auth_part}@{ip}:{port}/{path}"

def open_capture(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

@sio.event
def connect():
    print("[+] Успешно подключено к Node.js серверу!")

@sio.event
def disconnect():
    print("[-] Отключено от Node.js сервера!")

@sio.on("set_ai")
def set_ai(data):
    cam_id = int(data.get("cameraId"))
    ai_status[cam_id] = bool(data.get("enabled"))
    print(f"[*] Камера {cam_id} | Статус ИИ: {ai_status.get(cam_id)}")

@sio.on("init_cameras")
def on_init_cameras(cameras_list):
    print(f"[*] Получен список из {len(cameras_list)} камер от сервера.")
    for cam in cameras_list:
        on_start_camera_stream(cam)

@sio.on("start_camera_stream")
def on_start_camera_stream(data):
    cam_id = int(data["id"])
    if active_cameras.get(cam_id):
        return

    try:
        rtsp_url = build_rtsp_url(data)
    except Exception as exc:
        message = f"Ошибка настроек камеры: {exc}"
        print(f"[!] Камера {cam_id}: {message}")
        emit_camera_status(cam_id, False, "offline", message)
        return
    
    cam_info = {"id": cam_id, "url": rtsp_url}
    ai_status[cam_id] = data.get("aiEnabled", True)
    active_cameras[cam_id] = True
    
    t = threading.Thread(target=process_camera, args=(cam_info,))
    t.daemon = True
    t.start()

@sio.on("delete_camera")
def on_delete_camera(data):
    cam_id = int(data["id"])
    if cam_id in active_cameras:
        active_cameras[cam_id] = False 

def connect_to_server():
    while True:
        for node_url in SERVER_URLS:
            try:
                sio.connect(node_url, wait_timeout=5)
                return
            except Exception:
                pass
        time.sleep(3)

def process_camera(cam_info):
    cam_id = cam_info["id"]
    url = cam_info["url"]

    print(f"[*] AI Worker камеры {cam_id} запущен (Интервал {ANALYSIS_INTERVAL_SECONDS} сек).")

    last_analysis_time = 0
    was_online = False

    while active_cameras.get(cam_id, True):
        current_time = time.time()
        needs_analysis = ai_status.get(cam_id, True) and (current_time - last_analysis_time >= ANALYSIS_INTERVAL_SECONDS)

        if not needs_analysis:
            time.sleep(1)
            continue

        print(f"[>] Камера {cam_id}: Захват кадра для ИИ-анализа...")
        cap = open_capture(url)
        ret, frame = cap.read()
        cap.release() # Освобождаем девайс немедленно

        if not ret or frame is None:
            emit_camera_status(cam_id, False, "offline", "Нет сигнала")
            was_online = False
            time.sleep(5)
            continue

        if not was_online:
            emit_camera_status(cam_id, True, "online", "")
            was_online = True

        last_analysis_time = time.time()
        analysis_frame = cv2.resize(frame, FRAME_SIZE)
        
        results = model(analysis_frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=False, verbose=False, imgsz=640)
        
        status_priority = {"trash_around": 2, "bin_full": 1, "bin_empty": 0}
        global_trash_status = "unknown"
        max_priority = -1
        highest_confidence = 0.0
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD: continue
                
                cls = int(box.cls[0])
                class_name = str(model.names[cls]).lower()
                
                if "trash" in class_name or class_name == "2":
                    color, clean_name = (0, 0, 255), "trash_around"
                elif "full" in class_name or class_name == "1":
                    color, clean_name = (0, 165, 255), "bin_full"
                else:
                    color, clean_name = (0, 255, 0), "bin_empty"

                x1, y1, x2, y2 = tuple(map(int, box.xyxy[0]))
                cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(analysis_frame, f"{clean_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                priority = status_priority.get(clean_name, -1)
                if priority > max_priority:
                    max_priority = priority
                    global_trash_status = clean_name
                    highest_confidence = conf
                elif priority == max_priority and conf > highest_confidence:
                    highest_confidence = conf

        success, buffer = cv2.imencode(".jpg", analysis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if success:
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            sio.emit("camera_frame", {
                "type": "analysis",
                "cameraId": cam_id,
                "image": jpg_as_text,
                "trashStatus": global_trash_status,
                "confidence": highest_confidence,
                "fps": 0,
                "aiEnabled": ai_status.get(cam_id, True),
                "analysisIntervalSeconds": ANALYSIS_INTERVAL_SECONDS,
                "analyzedAt": int(time.time() * 1000)
            })
            print(f"[+] Камера {cam_id}: Анализ завершен. Ожидание {ANALYSIS_INTERVAL_SECONDS} сек.")

if __name__ == '__main__':
    connect_to_server()
    sio.wait()
