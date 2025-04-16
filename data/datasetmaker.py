import cv2
import os

# === Get path of "in" folder relative to script location ===
input_folder = os.path.join(os.path.dirname(__file__), "in")
output_folder = os.path.join(os.path.dirname(__file__), "out_faces")

# === Load Haar Cascade face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Check if input folder exists ===
if not os.path.exists(input_folder):
    print(f"[ERROR] Input folder '{input_folder}' not found.")
    exit()

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === Process all images in the folder ===
for i, filename in enumerate(os.listdir(input_folder)):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARNING] Skipping unreadable file: {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for j, (x, y, w, h) in enumerate(faces):
        face_crop = img[y:y + h, x:x + w]
        face_filename = os.path.join(output_folder, f"face_{i}_{j}.jpg")
        cv2.imwrite(face_filename, face_crop)
        print(f"[INFO] Saved: {face_filename}")

print("[DONE] All faces have been cropped and saved.")
