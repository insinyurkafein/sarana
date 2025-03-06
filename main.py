import cv2
import pyautogui
import pygame
import time
import os
import tkinter as tk
from tkinter import simpledialog
from ultralytics import YOLO

# Load Model YOLO
model = YOLO('yolov11.pt')

# Konstanta
FOCAL_LENGTH = 500  
REAL_EYE_WIDTH = 6  
THRESHOLD_DISTANCE = 50
WARNING_DURATION = 5
SCREEN_TIME_LIMIT = 300  # Default 5 menit, bisa diubah

# Inisialisasi Variabel
paused = False
start_warning_time = None
screen_time = 0  
last_eye_open_time = None  
last_update_time = time.time()  

# Inisialisasi Pygame untuk suara
pygame.mixer.init()
audio_safe_distance = pygame.mixer.Sound("safe_distance.mp3")
audio_screen_time = pygame.mixer.Sound("screen_time.mp3")

# Fungsi untuk memulai video setelah waktu diatur
def start_detection():
    global SCREEN_TIME_LIMIT
    try:
        SCREEN_TIME_LIMIT = int(entry.get()) * 60  # Konversi ke detik
        root.destroy()  # Tutup jendela tkinter setelah input diberikan
    except ValueError:
        label.config(text="Masukkan angka yang valid!", fg="red")

# GUI tkinter untuk input screen time
root = tk.Tk()
root.title("Set Screen Time")
root.geometry("300x150")

label = tk.Label(root, text="Atur waktu (menit):", font=("Arial", 12))
label.pack(pady=5)

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(pady=5)

button = tk.Button(root, text="Mulai", font=("Arial", 12), command=start_detection)
button.pack(pady=10)

root.mainloop()

# Buka Kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    eye_open_detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = result.names[class_id]

            if label == 'eye_open':
                eye_open_detected = True
                last_eye_open_time = time.time()

            if label in ['eye_open', 'eye_closed']:  
                eye_width_pixels = x2 - x1
                distance_cm = (REAL_EYE_WIDTH * FOCAL_LENGTH) / eye_width_pixels
                color = (0, 255, 0) if label == "eye_open" else (0, 0, 255)

                if distance_cm < THRESHOLD_DISTANCE:
                    if start_warning_time is None:
                        start_warning_time = time.time()

                    elapsed_time = time.time() - start_warning_time

                    if elapsed_time >= WARNING_DURATION and not paused:
                        pyautogui.press('space')
                        paused = True

                    if paused and not pygame.mixer.music.get_busy():
                        audio_safe_distance.play()

                else:
                    if paused:
                        pyautogui.press('space')
                        paused = False
                    
                    pygame.mixer.music.stop()
                    start_warning_time = None

                cv2.putText(frame, f"{distance_cm:.2f}cm", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    current_time = time.time()
    if eye_open_detected and (current_time - last_update_time >= 1):
        screen_time += 1
        last_update_time = current_time
    
    elif last_eye_open_time and (current_time - last_eye_open_time) > 1:
        last_eye_open_time = None

    cv2.putText(frame, f"Screen Time: {screen_time//3600:02}:{(screen_time%3600)//60:02}:{screen_time%60:02}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if screen_time >= SCREEN_TIME_LIMIT:
        audio_screen_time.play()
        time.sleep(10)
        os.system('taskkill /F /FI "STATUS eq RUNNING"')
        screen_time = 0  

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
