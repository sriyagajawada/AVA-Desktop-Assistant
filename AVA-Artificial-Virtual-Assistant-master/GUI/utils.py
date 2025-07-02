import requests
import wikipedia
import pywhatkit as kit
from email.message import EmailMessage
import smtplib
import gtts
from pydub import AudioSegment
import time
import threading
import keyboard
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import os
import pyautogui
import subprocess as sp
import webbrowser
import imdb
import cv2
from ultralytics import YOLO

yolo_model = YOLO("../yolo-weights/yolov8s.pt")  # Make sure the model file is in the correct path

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Set path to ffmpeg explicitly
AudioSegment.converter = r"C:\Users\sirignya reddy\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-full_build\ffmpeg-2025-03-31-git-35c091f4b7-full_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\sirignya reddy\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-full_build\ffmpeg-2025-03-31-git-35c091f4b7-full_build\bin\ffprobe.exe"

from decouple import config
from pydub import AudioSegment
from pydub.playback import play
from constants import (
    EMAIL,
    PASSWORD,
    IP_ADDR_API_URL,
    NEWS_FETCH_API_URL,
    WEATHER_FORECAST_API_URL,
    SMTP_URL,
    SMTP_PORT,
    NEWS_FETCH_API_KEY,
    WEATHER_FORECAST_API_KEY,
)


def speak(text):
    tts = gtts.gTTS(text, lang='en')
    tts.save("output.wav")

    audio = AudioSegment.from_file("output.wav")
    os.remove("output.wav")
    audio = audio.speedup(playback_speed=1.5)

    play(audio)


def find_my_ip():
    ip_address = requests.get('https://api64.ipify.org?format=json').json()
    return ip_address["ip"]


def search_on_wikipedia(query):
    results = wikipedia.summary(query, sentences=2)
    return results


def search_on_google(query):
    kit.search(query)


def youtube(video):
    query = video.replace(" ", "+")
    url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(url)


def send_email(receiver_add, subject, message):
    try:
        email = EmailMessage()
        email['To'] = receiver_add
        email['Subject'] = subject
        email['From'] = EMAIL

        email.set_content(message)
        s = smtplib.SMTP("smtp.gmail.com", 587)
        s.starttls()
        s.login("MAIL", "PASSKEY")
        s.send_message(email)
        s.close()
        return True

    except Exception as e:
        print(e)
        return False


def get_news():
    news_headline = []
    result = requests.get(
        "https://newsapi.org/v2/top-headlines?country=in&category=general",
        params={
            "country": "in",
            "category": "general",
            "apiKey": "NEWSAPI"
        },
    ).json()
    articles = result["articles"]
    for article in articles:
        news_headline.append(article["title"])
    return news_headline[:6]


def weather_forecast(city):
    res = requests.get(
        "https://api.openweathermap.org/data/2.5/weather?",
        params={
            "q": city,
            "appid": "WEATHERAPI",
            "units": "metric"
        },
    ).json()
    weather = res["weather"][0]["main"]
    temp = res["main"]["temp"]
    feels_like = res["main"]["feels_like"]
    return weather, f"{temp}°C", f"{feels_like}°C"


def run_object_detection():
    global object_detection_active, detected_objects

    # Initialize webcam with proper error handling
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Error: Could not open webcam.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm up camera
    for _ in range(5):
        cap.read()

    prev_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while object_detection_active:
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Read frame
        ret, frame = cap.read()
        if not ret:
            speak("Error reading frame from camera")
            break

        # Convert frame to RGB (YOLOv8 expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = yolo_model(rgb_frame, verbose=False)  # Disable logging for cleaner output

        # Reset current frame objects
        current_frame_objects = set()

        # Process detections
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Get confidence score
                conf = float(box.conf[0])
                if conf < 0.5:  # Skip low-confidence detections
                    continue

                # Get class ID and name
                cls_id = int(box.cls[0])
                cls_name = classNames[cls_id]
                current_frame_objects.add(cls_name)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Create label text
                label = f"{cls_name} {conf:.2f}"

                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(label, font, 0.7, 1)

                # Draw label background
                cv2.rectangle(frame, (x1, y1 - text_height - 10),
                              (x1 + text_width, y1), (0, 255, 0), -1)

                # Put text on frame
                cv2.putText(frame, label, (x1, y1 - 5), font, 0.7, (0, 0, 0), 2)

        # Check for new objects
        new_objects = current_frame_objects - detected_objects
        if new_objects:
            object_list = ", ".join(new_objects)
            speak(f"I detected {object_list}")
            detected_objects.update(new_objects)

        # Update detected objects
        detected_objects = current_frame_objects

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), font, 1, (0, 255, 0), 2)

        # Display detected objects count
        cv2.putText(frame, f"Objects: {len(detected_objects)}", (10, 70), font, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Object Detection - Press Q to quit", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detected_objects.clear()
    object_detection_active = False


def start_object_detection():
    global object_detection_active
    if not object_detection_active:
        object_detection_active = True
        detection_thread = threading.Thread(target=run_object_detection)
        detection_thread.daemon = True
        detection_thread.start()
        speak("Starting object detection")
    else:
        speak("Object detection is already running")


def stop_object_detection():
    global object_detection_active
    if object_detection_active:
        object_detection_active = False
        speak("Stopping object detection")
        if detected_objects:
            speak(f"I detected these objects: {', '.join(detected_objects)}")
    else:
        speak("Object detection is not active")