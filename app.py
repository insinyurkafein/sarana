from flask import Flask, render_template, request, redirect, send_file, url_for, Response
import cv2
import pyautogui
import pygame
import time
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov11.pt")  # Load model YOLO

pygame.mixer.init()
audio_safe_distance = pygame.mixer.Sound("safe_distance.mp3")
audio_screen_time = pygame.mixer.Sound("screen_time.mp3")

# Route untuk halaman input form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        screen_time = request.form['screen_time']
        # Redirect ke halaman arif dengan parameter screen_time
        return redirect(url_for('arif', screen_time=screen_time))
    return render_template('screen_time.html')

# Route untuk halaman arif.html
@app.route('/arif')
def arif():
    screen_time = request.args.get('screen_time', default=300, type=int)
    return render_template('arif.html', screen_time=screen_time)

def generate_frames():
    FOCAL_LENGTH = 500  
    REAL_EYE_WIDTH = 6  
    THRESHOLD_DISTANCE = 50
    WARNING_DURATION = 5

    paused = False
    start_warning_time = None
    screen_time = 0  
    last_eye_open_time = None  
    last_update_time = time.time()  
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
                label = result.names[int(box.cls[0])]

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

                        if paused and audio_safe_distance.get_num_channels() == 0:
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

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

    
@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')

@app.route('/sw.js')
def serve_sw():
    return send_file('sw.js', mimetype='application/javascript')