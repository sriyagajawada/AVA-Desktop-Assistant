import time
import threading
import keyboard
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import os
import pyautogui
from datetime import datetime

from pymongo import MongoClient
import pywhatkit
import subprocess as sp
import webbrowser
import imdb
import cv2
import base64
import sqlite3
from kivy.app import App
from kivy.properties import BooleanProperty, StringProperty, NumericProperty
from kivy.uix.widget import Widget
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.animation import Animation
from constants import SCREEN_HEIGHT, SCREEN_WIDTH, GEMINI_API_KEY
from utils import (speak, youtube, search_on_google, search_on_wikipedia,
                   send_email, get_news, weather_forecast, classNames,
                   yolo_model, find_my_ip)
import google.generativeai as genai


# Database Configuration
class ChatDatabase:
    def __init__(self):
        self.db_type = self._detect_db_type()
        self.__init__db()

    def _detect_db_type(self, pymongo=None):
        """Try MongoDB first, fallback to SQLite"""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure
            client = MongoClient("mongodb://localhost:27017")
            client.admin.command('ismaster')
            print("Using MongoDB for storage")
            return "mongodb"
        except Exception as e:
            print(f"Could not connect to MongoDB, using SQLite instead: {e}")
            return "sqlite"

    def __init__db(self):
        """Initialize the appropriate database"""
        if self.db_type == "mongodb":
            from pymongo import MongoClient
            self.client = MongoClient("mongodb://localhost:27017/")
            self.db = self.client["ava_chat"]
            self.chats = self.db["conversations"]
        else:
            self.conn = sqlite3.connect('ava_chat.db')
            self.c = self.conn.cursor()
            self.c.execute('''CREATE TABLE IF NOT EXISTS chats
                           (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            sender TEXT,
                            content TEXT,
                            media_type TEXT,
                            media_data BLOB)''')
            self.conn.commit()

    def save_message(self, content, sender, media_type=None, media_data=None):
        """Save message to database"""
        print(f"Saving message using {self.db_type}")  # Add this line
        timestamp = datetime.now().isoformat()

        if self.db_type == "mongodb":
            self.chats.insert_one({
                "timestamp": timestamp,
                "sender": sender,
                "content": content,
                "media_type": media_type,
                "media_data": media_data
            })
        else:
            if media_data and isinstance(media_data, bytes):
                media_data = sqlite3.Binary(media_data)
            self.c.execute("INSERT INTO chats VALUES (NULL, ?, ?, ?, ?, ?)",
                           (timestamp, sender, content, media_type, media_data))
            self.conn.commit()

    def get_history(self, limit=100):
        """Retrieve chat history"""
        if self.db_type == "mongodb":
            return list(self.chats.find().sort("timestamp", 1).limit(limit))
        else:
            self.c.execute("SELECT * FROM chats ORDER BY timestamp ASC LIMIT ?", (limit,))
            return self.c.fetchall()

    def close(self):
        """Close database connection"""
        if self.db_type == "mongodb":
            self.client.close()
        else:
            self.conn.close()


# Initialize Database
db = ChatDatabase()

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Object Detection Globals
object_detection_active = False
detection_thread = None
detected_objects = set()


class RoundedButton(Button):
    def __init__(self, **kwargs):
        super(RoundedButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0.2, 0.6, 1, 1)
        self.color = (1, 1, 1, 1)
        self.size_hint = (None, None)
        self.size = (100, 50)  # Set a default size
        with self.canvas.before:
            Color(*self.background_color)
            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[15]
            )
        self.bind(
            pos=self._update_rect,
            size=self._update_rect,
            state=self._on_state
        )

    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def _on_state(self, instance, value):
        # Change color when pressed
        if value == 'down':
            self.background_color = (0.1, 0.5, 0.9, 1)
        else:
            self.background_color = (0.2, 0.6, 1, 1)


class MessageBubble(BoxLayout):
    text = StringProperty('')
    is_user = BooleanProperty(False)

    def __init__(self, text, is_user=False, **kwargs):
        super(MessageBubble, self).__init__(**kwargs)
        self.text = text
        self.is_user = is_user
        self.orientation = 'horizontal'
        self.size_hint = (None, None)

        # Calculate width and height
        bubble_width = min(300, max(100, len(text) // 25 * 100))  # enforce minimum width
        bubble_height = max(40, len(text) // 25 * 20)

        self.width = bubble_width
        self.height = bubble_height
        self.pos_hint = {'right': 1} if is_user else {'x': 0}
        self.padding = [10, 5]
        self.spacing = 5

        with self.canvas.before:
            Color(*((0.2, 0.6, 1, 1) if is_user else (0.9, 0.9, 0.9, 1)))
            self.rect = RoundedRectangle(
                size=self.size,
                pos=self.pos,
                radius=[15]
            )

        self.bind(size=self._update_rect, pos=self._update_rect)

        self.label = Label(
            text=text,
            color=(1, 1, 1, 1) if is_user else (0, 0, 0, 1),
            size_hint=(None, None),
            size=(self.width, self.height),
            halign='left',
            valign='middle',
            text_size=(self.width - 20, None)
        )
        self.label.bind(texture_size=self._adjust_height)
        self.add_widget(self.label)

    def _update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def _adjust_height(self, instance, value):
        """Adjust bubble height based on label content."""
        self.label.height = value[1]
        self.height = self.label.height + 10


class Ava(FloatLayout):
    circle_size = NumericProperty(100)

    def __init__(self, **kwargs):
        super(Ava, self).__init__(**kwargs)
        # Initialize object detection variables
        self.object_detection_active = False
        self.detection_thread = None
        self.detected_objects = set()
        self.cap = None

        # Circle visualization
        with self.canvas.before:
            Color(0.2, 0.6, 1, 0.2)  # Background circle
            self.bg_circle = Ellipse(size=(self.circle_size, self.circle_size))
            Color(0.2, 0.6, 1, 0.7)  # Main circle
            self.circle = Ellipse(size=(self.circle_size, self.circle_size))

        self.bind(pos=self.update_circle_pos, size=self.update_circle_pos)

        # Volume properties
        self.volume = 0
        self.volume_history = [0] * 7
        self.volume_history_size = 140
        self.min_size = 50
        self.max_size = 300

        # Main UI components
        self.setup_ui()

        # Start services
        self.load_chat_history()
        self.start_listening()
        Clock.schedule_interval(self.update_circle, 1 / 60)
        keyboard.add_hotkey('`', self.start_recording)

    def setup_ui(self):
        """Initialize all UI components"""
        # Main layout
        self.main_layout = BoxLayout(orientation='vertical')
        self.add_widget(self.main_layout)

        # Header
        self.header = BoxLayout(size_hint=(1, 0.08), padding=[10, 5])
        self.header.add_widget(Label(
            text='AVA',
            font_size=24,
            color=(0.2, 0.6, 1, 1),
            bold=True
        ))
        self.main_layout.add_widget(self.header)

        # Chat area
        self.chat_area = ScrollView(size_hint=(1, 0.82))
        self.chat_history = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=15,
            padding=[20, 10, 20, 20]
        )
        self.chat_history.bind(minimum_height=self.chat_history.setter('height'))
        self.chat_area.add_widget(self.chat_history)
        self.main_layout.add_widget(self.chat_area)

        # Input area
        self.input_layout = BoxLayout(
            size_hint=(1, 0.1),
            spacing=15,
            padding=[20, 10, 20, 10]
        )

        # Voice button
        self.voice_btn = RoundedButton(
            text="🎤",
            size_hint=(0.15, 1)
        )
        self.voice_btn.bind(on_press=self.start_recording)

        # Text input
        self.chat_input = TextInput(
            hint_text="Type your message...",
            size_hint=(0.7, 1),
            background_normal='',
            background_color=(0.95, 0.95, 0.95, 1),
            foreground_color=(0, 0, 0, 1),
            padding=[10, 10]
        )

        # Send button
        self.send_btn = RoundedButton(
            text="Send",
            size_hint=(0.15, 1)
        )
        self.send_btn.bind(on_release=self.send_text_message)

        self.input_layout.add_widget(self.voice_btn)
        self.input_layout.add_widget(self.chat_input)
        self.input_layout.add_widget(self.send_btn)
        self.main_layout.add_widget(self.input_layout)

    def update_circle_pos(self, *args):
        """Update circle positions"""
        self.circle.pos = (
            self.center_x - self.circle_size / 2,
            self.center_y - self.circle_size / 2
        )
        self.bg_circle.pos = (
            self.center_x - self.max_size / 2,
            self.center_y - self.max_size / 2
        )
        self.bg_circle.size = (self.max_size, self.max_size)

    def update_circle(self, dt):
        """Animate circle based on volume"""
        try:
            self.size_value = int(np.mean(self.volume_history))
            self.size_value = max(self.min_size, min(self.max_size, self.size_value))

            # Animate the size change
            anim = Animation(
                circle_size=self.size_value,
                duration=0.1,
                t='out_quad'
            )
            anim.start(self)

        except Exception as e:
            print('Warning:', e)
            self.circle_size = self.min_size

    def add_message_bubble(self, text, is_user=False, media_path=None):
        """Add a styled message bubble to the chat"""
        bubble = MessageBubble(text, is_user)
        self.chat_history.add_widget(bubble)

        if media_path:
            img = KivyImage(
                source=media_path,
                size_hint=(None, None),
                size=(300, 200),
                pos_hint={'right': 1} if is_user else {'x': 0},
                allow_stretch=True
            )
            self.chat_history.add_widget(img)

        self.chat_area.scroll_y = 0

    def load_chat_history(self):
        """Load previous chats with new bubble style"""
        try:
            history = db.get_history()
            for chat in history:
                media_path = ""
                if chat.get("media_type") == "image":
                    img_data = chat["media_data"]
                    if db.db_type == "sqlite":
                        img_data = bytes(img_data)
                    media_path = f"temp_img_{chat['timestamp']}.jpg"
                    with open(media_path, "wb") as f:
                        f.write(base64.b64decode(img_data))

                self.add_message_bubble(
                    chat["content"],
                    is_user=chat["sender"] == "user",
                    media_path=media_path
                )
        except Exception as e:
            print(f"Error loading chat history: {e}")

    def send_text_message(self, instance):
        """Handle text message sending"""
        text = self.chat_input.text.strip()
        if text:
            if hasattr(self, 'weather_query_active') and self.weather_query_active:
                # Handle weather city input
                self.process_weather_request(text)
                self.weather_query_active = False
                self.chat_input.hint_text = "Type your message..."
                return

            # Normal message handling
            self.add_message_bubble(text, is_user=True)
            db.save_message(text, "user")
            self.chat_input.text = ""

            # Process command in separate thread
            threading.Thread(target=self.process_command, args=(text,)).start()

    def process_command(self, text):
        """Process command in a background thread"""
        try:
            # First try to handle as a specific command
            response = self.handle_command(text)

            # If no specific command matched, use Gemini
            if response is None:
                response = self.get_gemini_response(text)
                response = response.replace("*", "")

            # Update UI on main thread
            Clock.schedule_once(lambda dt: (
                self.add_message_bubble(response, is_user=False),
                db.save_message(response, "assistant"),
                speak(response)
            ))

        except Exception as e:
            print(f"Error processing command: {e}")
            Clock.schedule_once(lambda dt: self.add_message_bubble(
                "Sorry, I encountered an error processing your request",
                is_user=False
            ))

    def start_recording(self, *args):
        print("Starting voice recording...")
        threading.Thread(target=self.run_speech_recognition).start()

    def run_speech_recognition(self):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("Calibrating mic...")
                r.adjust_for_ambient_noise(source, duration=1)
                print("Listening...")
                Clock.schedule_once(lambda dt: setattr(self.chat_input, 'text', "Listening..."))
                audio = r.listen(source, timeout=5)
                print("Audio recorded")

            query = r.recognize_google(audio, language="en-in")
            print(f'Recognized: {query}')

            # Update UI and process command
            Clock.schedule_once(lambda dt: (
                setattr(self.chat_input, 'text', query),
                self.process_command(query.lower())
            ))

        except sr.UnknownValueError:
            print("Could not understand audio")
            Clock.schedule_once(lambda dt: (
                setattr(self.chat_input, 'text', "Could not understand audio"),
                self.add_message_bubble("Could not understand audio", is_user=False)
            ))

        except sr.RequestError as e:
            print(f"Google API error: {e}")
            Clock.schedule_once(lambda dt: (
                setattr(self.chat_input, 'text', "Connection error"),
                self.add_message_bubble("Connection error with speech service", is_user=False)
            ))

        except Exception as e:
            print(f"General error: {e}")
            Clock.schedule_once(lambda dt: (
                setattr(self.chat_input, 'text', "Error in voice recognition"),
                self.add_message_bubble("Error in voice recognition", is_user=False)
            ))

    def update_volume(self, indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 200
        self.volume = volume_norm
        self.volume_history.append(volume_norm)

        if len(self.volume_history) > self.volume_history_size:
            self.volume_history.pop(0)

    def start_listening(self):
        self.stream = sd.InputStream(callback=self.update_volume)
        self.stream.start()

    def get_gemini_response(self, query):
        try:
            response = gemini_model.generate_content(query)
            return response.text
        except Exception as e:
            print(f"Error getting Gemini response: {e}")
            return "I'm sorry, I couldn't process that request."

    def init_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")

            # Camera setup
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            Clock.schedule_once(lambda dt: self.add_message_bubble(
                "Error: Could not open webcam", is_user=False
            ))
            return False

    def release_camera(self):
        """Release camera resources"""
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def run_object_detection(self):
        """Run YOLO object detection on webcam feed"""
        try:
            if not self.init_camera():
                return

            start_time = time.time()
            timeout = 45  # 45 second timeout

            while self.object_detection_active and (time.time() - start_time < timeout):
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Perform detection
                results = yolo_model(frame, verbose=False)

                # Process detections
                current_frame_objects = set()
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf < 0.5:
                            continue

                        cls_id = int(box.cls[0])
                        cls_name = classNames[cls_id]
                        current_frame_objects.add(cls_name)

                # Announce new objects
                new_objects = current_frame_objects - self.detected_objects
                if new_objects:
                    object_list = ", ".join(new_objects)
                    Clock.schedule_once(lambda dt: speak(f"I detected {object_list}"))
                    self.detected_objects.update(new_objects)

            # Cleanup
            self.release_camera()

            # Report results
            if time.time() - start_time >= timeout:
                Clock.schedule_once(lambda dt: speak("Object detection completed after 45 seconds"))
            if self.detected_objects:
                object_list = ', '.join(self.detected_objects)
                Clock.schedule_once(lambda dt: speak(f"I detected these objects: {object_list}"))
                print("Detected Objects:", object_list)
            self.detected_objects.clear()

        except Exception as e:
            print(f"Error in object detection: {e}")
            Clock.schedule_once(lambda dt: speak("There was an error in object detection"))
        finally:
            self.object_detection_active = False
            self.release_camera()

    def start_object_detection(self):
        """Start object detection with 45-second timeout"""
        if not self.object_detection_active:
            self.object_detection_active = True
            self.detected_objects.clear()

            # Start detection thread
            self.detection_thread = threading.Thread(target=self.run_object_detection)
            self.detection_thread.daemon = True
            self.detection_thread.start()

            # Update UI immediately
            Clock.schedule_once(lambda dt: (
                self.add_message_bubble(
                    "Starting object detection for 45 seconds",
                    is_user=False
                ),
                speak("Starting object detection for 45 seconds")
            ))
        else:
            Clock.schedule_once(lambda dt: (
                self.add_message_bubble(
                    "Object detection is already running",
                    is_user=False
                ),
                speak("Object detection is already running")
            ))

    def stop_object_detection(self):
        """Stop object detection"""
        if self.object_detection_active:
            self.object_detection_active = False
            if self.detected_objects:
                object_list = ', '.join(self.detected_objects)
                Clock.schedule_once(lambda dt: (
                    self.add_message_bubble(
                        f"Stopping object detection. Detected objects: {object_list}",
                        is_user=False
                    ),
                    speak(f"Stopping object detection. I detected these objects: {object_list}")
                ))
                print("Detected Objects:", object_list)
            else:
                Clock.schedule_once(lambda dt: (
                    self.add_message_bubble(
                        "Stopping object detection. No objects detected",
                        is_user=False
                    ),
                    speak("Stopping object detection. No objects detected")
                ))

            # Release camera resources
            self.release_camera()
        else:
            Clock.schedule_once(lambda dt: (
                self.add_message_bubble(
                    "Object detection is not active",
                    is_user=False
                ),
                speak("Object detection is not active")
            ))

    def process_weather_request(self, city):
        """Process weather request for a specific city"""

        def weather_thread():
            try:
                # Get weather data
                weather, temp, feels_like = weather_forecast(city)

                # Prepare response
                response = (f"Weather report for {city}:\n"
                            f"Description: {weather}\n"
                            f"Temperature: {temp}\n"
                            f"Feels like: {feels_like}")

                # Update UI on main thread
                Clock.schedule_once(lambda dt: (
                    self.add_message_bubble(response, is_user=False),
                    db.save_message(response, "assistant"),
                    speak(response)
                ))

            except Exception as e:
                print(f"Error getting weather: {e}")
                Clock.schedule_once(lambda dt: (
                    self.add_message_bubble("Sorry, I couldn't get the weather information", is_user=False),
                    speak("Sorry, I couldn't get the weather information")
                ))

        # Start weather lookup in background thread
        threading.Thread(target=weather_thread, daemon=True).start()

    def process_news_request(self):
        """Fetch and display news headlines"""
        try:
            # Get news from your utils function
            news_items = get_news()

            # Format the news response
            if isinstance(news_items, list):
                response = "Here are the latest news headlines:\n\n" + "\n\n".join(news_items[:5])  # Show top 5 news
            else:
                response = f"Latest news: {news_items}"

            # Update UI on main thread
            Clock.schedule_once(lambda dt: (
                self.add_message_bubble(response, is_user=False),
                db.save_message(response, "assistant"),
                speak("Here are the latest news headlines")
            ))

        except Exception as e:
            print(f"Error getting news: {e}")
            error_msg = "Sorry, I couldn't fetch the latest news"
            Clock.schedule_once(lambda dt: (
                self.add_message_bubble(error_msg, is_user=False),
                speak(error_msg)
            ))

    def handle_command(self, query):
        """Handle specific commands before falling back to Gemini"""
        query = query.lower()

        # Basic commands
        if "stop object detection" in query or "what do you see" in query:
            self.stop_object_detection()
            return None  # Prevent Gemini from processing

        elif "start object detection" in query or "detect objects" in query:
            self.start_object_detection()
            return None  # Prevent Gemini from processing

        elif "list detected objects" in query:
            if self.detected_objects:
                object_list = ', '.join(self.detected_objects)
                return f"Detected objects: {object_list}"
            return "No objects detected yet. Start object detection first."

        elif any(greeting in query for greeting in ['hello', 'hi', 'hey']):
            return "Hello! How can I assist you today?"

        elif 'how are you' in query:
            return "I'm functioning optimally. How can I help you?"

        elif 'thank you' in query or 'thanks' in query:
            return "You're welcome! Is there anything else I can do for you?"

        # System commands
        elif 'open command prompt' in query or 'open cmd' in query:
            os.system('start cmd')
            return "Command prompt opened"

        elif 'open camera' in query:
            sp.run('start microsoft.windows.camera:', shell=True)
            return "Camera app opened"

        elif 'open notepad' in query:
            os.startfile("notepad.exe")
            return "Notepad opened"

        elif 'ip address' in query:
            return f"Your IP address is {find_my_ip()}"

        # Web commands
        elif 'youtube' in query:
            if 'play' in query:
                video = query.replace('play', '').replace('on youtube', '').replace('youtube', '').strip()
            else:
                video = query.replace('youtube', '').strip()

            if video:
                video = '+'.join(video.split())
                webbrowser.open(f"https://www.youtube.com/results?search_query={video}")
                return f"Searching YouTube for {video.replace('+', ' ')}"
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube"

        elif 'google' in query or 'search' in query:
            if 'search' in query:
                search_term = query.replace('search', '').replace('on google', '').replace('google', '').strip()
            else:
                search_term = query.replace('google', '').strip()

            if search_term:
                search_term = '+'.join(search_term.split())
                webbrowser.open(f"https://www.google.com/search?q={search_term}")
                return f"Searching Google for {search_term.replace('+', ' ')}"
            webbrowser.open("https://www.google.com")
            return "Opening Google"

        elif 'search on wikipedia' in query or 'wikipedia' in query:
            search_term = query.replace('search on wikipedia', '').replace('wikipedia', '').strip()
            if search_term:
                result = search_on_wikipedia(search_term)
                return f"According to Wikipedia: {result}"
            return "What would you like to search on Wikipedia?"

        # Communication commands
        elif 'send email' in query:
            try:
                parts = query.split('to')[1].split('subject')
                receiver = parts[0].strip()
                subject_part = parts[1].split('message')
                subject = subject_part[0].strip()
                message = subject_part[1].strip() if len(subject_part) > 1 else ""

                if send_email(receiver, subject, message):
                    return f"Email sent to {receiver}"
                return "Failed to send email"
            except Exception as e:
                return f"Please use format: 'send email to [address] subject [subject] message [message]'"

        elif 'whatsapp' in query:
            try:
                parts = query.split('to')[1].split('message')
                contact = parts[0].strip()
                message = parts[1].strip()

                contacts = {
                    "amma": "+phno",
                    "nanna": "+phno",
                    "siri": "+phno"
                }

                if contact.lower() in contacts:
                    number = contacts[contact.lower()]
                    now = datetime.now()
                    pywhatkit.sendwhatmsg_instantly(number, message, wait_time=15)
                    return f"Message sent to {contact}"
                return f"Contact {contact} not found"
            except Exception as e:
                return f"Please use format: 'send whatsapp message to [contact] message [message]'"

        # Information commands
        elif 'news' in query:
            # Process news in a background thread
            threading.Thread(target=self.process_news_request, daemon=True).start()
            return "Fetching the latest news headlines..."

        elif 'weather' in query:
            # First check if city is already in query
            if ' in ' in query:
                city = query.split(' in ')[-1].strip()
                self.process_weather_request(city)
                return None  # Don't send additional response

            # Otherwise prompt for city
            def ask_for_city(dt):
                self.chat_input.text = ""
                self.chat_input.hint_text = "Please enter your city name"
                self.weather_query_active = True

            Clock.schedule_once(ask_for_city)
            return "Which city's weather would you like to know? Please enter it below."

        elif "movie" in query:
            movies_db = imdb.IMDb()
            speak("Please tell me the movie name:")
            text = self.take_command()
            movies = movies_db.search_movie(text)
            speak("searching for" + text)
            speak("I found these")
            for movie in movies:
                title = movie["title"]
                year = movie["year"]
                speak(f"{title}-{year}")
                info = movie.getID()
                movie_info = movies_db.get_movie(info)
                rating = movie_info["rating"]
                cast = movie_info["cast"]
                actor = cast[0:5]
                plot = movie_info.get('plot outline', 'plot summary not available')
                speak(f"{title} was released in {year} has imdb ratings of {rating}.It has a cast of {actor}. "
                      f"The plot summary of movie is {plot}")
                print(f"{title} was released in {year} has imdb ratings of {rating}.\n It has a cast of {actor}. \n"
                      f"The plot summary of movie is {plot}")

        else:
            gemini_response = self.get_gemini_response(query)
            gemini_response = gemini_response.replace("*", "")
            if gemini_response and gemini_response != "I'm sorry, I couldn't process that request.":
                speak(gemini_response)
                print(gemini_response)


class AvaApp(App):
    def build(self):
        return Ava()

    def on_stop(self):
        # Clean up resources when app stops
        if hasattr(self.root, 'stream') and self.root.stream:
            self.root.stream.stop()
            self.root.stream.close()

        # Ensure camera is released
        if hasattr(self.root, 'cap') and self.root.cap and self.root.cap.isOpened():
            self.root.cap.release()

        return True


# Run the application
if __name__ == '__main__':
    AvaApp().run()

