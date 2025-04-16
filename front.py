import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import os
import pickle
import numpy as np
import pygame
import csv
from datetime import datetime

# --- Configuration ---
CACHE_FILE = "face_db.pkl"
CSV_FILE = "surveillance_log.csv"

# --- Initialisation ---
pygame.mixer.init()
yolo_model = YOLO("yolov8n.pt")
try:
    alarm_sound = pygame.mixer.Sound("al.mp3")
except:
    alarm_sound = None

# --- DeepFace DB ---
def build_face_db(db_path="known_faces/known_faces/", save_cache=True):
    face_db = []
    for person in os.listdir(db_path):
        for img in os.listdir(os.path.join(db_path, person)):
            try:
                full_path = os.path.join(db_path, person, img)
                emb = DeepFace.represent(img_path=full_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                face_db.append({"name": person, "embedding": emb})
            except Exception as e:
                print("Erreur DeepFace:", e)
    if save_cache:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(face_db, f)
    return face_db

def load_face_db():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return None

def recognize_face(temp_path, face_db, threshold=10.0):
    try:
        emb = DeepFace.represent(img_path=temp_path, model_name="Facenet", enforce_detection=True)[0]["embedding"]
        closest = "Unknown"
        min_dist = float("inf")
        for entry in face_db:
            dist = np.linalg.norm(np.array(emb) - np.array(entry["embedding"]))
            if dist < min_dist:
                min_dist = dist
                closest = entry["name"]
        return closest if min_dist < threshold else "Unknown"
    except:
        return "Unknown"

# --- CSV Logging ---
def log_event(name, place, event):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, time, place, event])

# --- Interface Surveillance ---
class SurveillanceApp:
    def __init__(self, face_db):
        self.face_db = face_db
        self.alert_active = False
        self.previous_zone = {}

        self.root = tk.Tk()
        self.root.title("Surveillance Piscine")

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = yolo_model(frame)
        frame = self.process_detections(frame, results)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.video_label.after(10, self.update_frame)

    def process_detections(self, frame, results):
        pool_alert = False

        # Zones with fixed sizes
        house_zone = (0, 0, 300, 400)
        pool_zone = (300, 0, 600, 400)

        # Draw zones
        cv2.rectangle(frame, house_zone[:2], house_zone[2:], (0, 255, 255), 2)
        cv2.putText(frame, "House", (house_zone[0]+10, house_zone[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.rectangle(frame, pool_zone[:2], pool_zone[2:], (0, 0, 255), 2)
        cv2.putText(frame, "Pool", (pool_zone[0]+10, pool_zone[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                cv2.imwrite("temp.jpg", face_crop)
                name = recognize_face("temp.jpg", self.face_db)

                cx = (x1 + x2) // 2
                zone = "Unknown"
                color = (255, 255, 255)

                if house_zone[0] <= cx <= house_zone[2]:
                    zone = "House"
                    color = (0, 255, 0)
                elif pool_zone[0] <= cx <= pool_zone[2]:
                    zone = "Pool"
                    color = (0, 0, 255)
                    if "baha" in name.lower():
                        pool_alert = True

                label = f"{name} ({zone})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Entry/Exit detection
                if name != "Unknown":
                    if name not in self.previous_zone:
                        self.previous_zone[name] = None

                    if self.previous_zone[name] != zone:
                        if self.previous_zone[name] is not None:
                            event = "Entered" if zone == "Pool" and self.previous_zone[name] == "House" else "Left"
                            log_event(name, zone, event)
                        self.previous_zone[name] = zone

        if pool_alert and not self.alert_active:
            if alarm_sound:
                alarm_sound.play(-1)
            self.alert_active = True
        elif not pool_alert and self.alert_active:
            if alarm_sound:
                alarm_sound.stop()
            self.alert_active = False


        return frame

    def on_close(self):
        if alarm_sound:
            alarm_sound.stop()
        self.cap.release()
        self.root.destroy()

# --- Démarrage ---
if __name__ == "__main__":
    print("1. Charger la base de visages")
    print("2. Recréer depuis le dossier known_faces/")

    choice = input("Choix (1 ou 2) : ").strip()
    if choice == "1":
        db = load_face_db()
        if not db:
            db = build_face_db()
    else:
        db = build_face_db()

    SurveillanceApp(db)
