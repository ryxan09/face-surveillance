from ultralytics import YOLO
from deepface import DeepFace
import cv2
import os
import numpy as np
import time
import pickle

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# File to save the face embeddings!
CACHE_FILE = "face_db.pkl"

# Build face database and optionally save
def build_face_db(db_path="known_faces/known_faces/", save_cache=True):
    face_db = []
    for person in os.listdir(db_path):
        person_dir = os.path.join(db_path, person)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                face_db.append({"name": person, "embedding": embedding})
                print(f"[INFO] Added {person}/{img_name} to DB")
            except Exception as e:
                print(f"[ERROR] Could not process {img_path}: {e}")

    if save_cache:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(face_db, f)
        print("[INFO] Face database saved to cache.")

    return face_db

# Load saved face database
def load_face_db():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            print("[INFO] Loaded face database from cache.")
            return pickle.load(f)
    else:
        print("[WARN] Cache not found.")
        return None

# Compare face with known embeddings
def recognize_face(face_img_array, face_db, threshold=10.0):
    try:
        embedding = DeepFace.represent(img_path=face_img_array, model_name="Facenet", enforce_detection=True)[0]["embedding"]

        closest_match = "Unknown"
        min_distance = float("inf")

        for entry in face_db:
            dist = np.linalg.norm(np.array(embedding) - np.array(entry["embedding"]))
            print(f"[DEBUG] Compared with {entry['name']}: distance = {dist:.2f}")
            if dist < min_distance:
                min_distance = dist
                closest_match = entry["name"]

        print(f"[DEBUG] Closest match: {closest_match}, distance: {min_distance:.2f}")
        if min_distance < threshold:
            return closest_match
    except Exception as e:
        print(f"[ERROR] DeepFace issue: {e}")
    return "Unknown"

# Real-time face recognition with FPS
def recognize_from_video(face_db):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam not detected.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)

        for r in results:
            if not r.boxes:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0 or (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                temp_path = "temp_face.jpg"
                cv2.imwrite(temp_path, face_crop)

                name = recognize_face(temp_path, face_db)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    print("Do you want to load the saved face database or rebuild it?")
    print("1. Load from saved")
    print("2. Rebuild from images")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        face_db = load_face_db()
        if not face_db:
            print("[INFO] Rebuilding because loading failed...")
            face_db = build_face_db()
    else:
        face_db = build_face_db()

    print("[INFO] Starting real-time recognition...")
    recognize_from_video(face_db)
