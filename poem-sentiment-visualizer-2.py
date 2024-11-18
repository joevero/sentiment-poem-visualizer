import cv2
import numpy as np
import random
import time
from typing import List, Tuple
from fer import FER
import requests
from io import BytesIO

class EmotionalPoemGenerator:
    def __init__(self):
        self.happy_poems = [
            [
                "Sunshine in your smile",
                "Keep that joy burning bright",
                "Light up the world"
            ],
            [
                "Your happiness glows",
                "Like stars dancing in night",
                "Never let it fade"
            ],
            [
                "Joy flows from within",
                "A beacon of hope and light",
                "Stay radiant now"
            ]
        ]

        self.uplifting_poems = [
            [
                "After the rainfall",
                "Rainbow colors paint the sky",
                "Hope blooms anew"
            ],
            [
                "Dark clouds will pass by",
                "Sun breaks through the grey",
                "Your strength guides way"
            ],
            [
                "Each tear that falls down",
                "Waters seeds of tomorrow",
                "New joy will spring"
            ]
        ]

        self.neutral_poems = [
            [
                "Mirror reflects truth",
                "Silver and exact, unmoved",
                "Time passes, fades"
            ],
            [
                "Between light and dark",
                "Moon's face tells no stories",
                "Silence speaks now"
            ],
            [
                "Thoughts spiral like moths",
                "Around the flame of being",
                "Neither here there"
            ]
        ]

    def get_poem(self, emotion: str) -> Tuple[List[str], str]:
        if emotion == "happy":
            return (
                random.choice(self.happy_poems),
                "You're feeling happy! Here's a motivating poem!"
            )
        elif emotion == "sad":
            return (
                random.choice(self.uplifting_poems),
                "You seem a bit down ): here's something to cheer you up!"
            )
        else:  # neutral
            return (
                random.choice(self.neutral_poems),
                "You seem oddly neutral.. here's something to think about."
            )

class EmotionalWebcamPoetry:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.emotion_detector = FER(mtcnn=True)
        self.poem_generator = EmotionalPoemGenerator()
        
        self.current_poem = None
        self.current_message = ""
        self.poem_timestamp = time.time()
        self.poem_refresh_interval = 3
        
        # frame overlay placeholder
        self.frame_overlay = None
        self.load_frame_overlay('https://www.pngall.com/wp-content/uploads/4/Vintage-Frame-Transparent.png')

    def load_frame_overlay(self, url: str):
        try:
            response = requests.get(url)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            self.frame_overlay = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Error loading frame overlay: {e}")
            self.frame_overlay = None

    def add_text_with_background(self, frame, text, position, font_scale, color, thickness=2):
        """Add text with a dark background for better visibility"""
        font = cv2.FONT_HERSHEY_COMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Add dark background rectangle
        padding = 10
        start_point = (position[0] - padding, position[1] - text_size[1] - padding)
        end_point = (position[0] + text_size[0] + padding, position[1] + padding)
        cv2.rectangle(frame, start_point, end_point, (0, 0, 0), cv2.FILLED)
        
        # Add text
        cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    def overlay_frame(self, image: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        if self.frame_overlay is None:
            x, y, w, h = face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            return image
        
        x, y, w, h = face
        padding = int(w * 0.4)
        frame_w = w + 2 * padding
        frame_h = h + 2 * padding
        frame_x = max(0, x - padding)
        frame_y = max(0, y - padding)
        
        try:
            if self.frame_overlay is not None:
                resized_frame = cv2.resize(self.frame_overlay, (frame_w, frame_h))
                if resized_frame.shape[2] == 4:  # If has alpha channel
                    alpha = resized_frame[:, :, 3] / 255.0
                    for c in range(3):
                        image[frame_y:frame_y+frame_h, frame_x:frame_x+frame_w, c] = \
                            image[frame_y:frame_y+frame_h, frame_x:frame_x+frame_w, c] * (1 - alpha) + \
                            resized_frame[:, :, c] * alpha
            
        except Exception as e:
            print(f"Error overlaying frame: {e}")
            
        return image

    def detect_emotion(self, frame: np.ndarray) -> str:
        emotions = self.emotion_detector.detect_emotions(frame)
        
        if emotions:
            emotions_dict = emotions[0]['emotions']
            if emotions_dict['happy'] > 0.4:
                return "happy"
            elif emotions_dict['sad'] > 0.3:
                return "sad"
            else:
                return "neutral"
        
        return "neutral"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        current_time = time.time()
        if current_time - self.poem_timestamp > self.poem_refresh_interval:
            if len(faces) > 0:
                emotion = self.detect_emotion(frame)
                self.current_poem, self.current_message = self.poem_generator.get_poem(emotion)
            self.poem_timestamp = current_time

        for face in faces:
            x, y, w, h = face
            frame = self.overlay_frame(frame, face)
            
            # adddd emotion message above faceq
            message_scale = 0.65  # increased scale for emotion message
            message_y = y - 10  # psition just above the face box
            self.add_text_with_background(frame, self.current_message, 
                                        (x, message_y), message_scale, 
                                        (255, 255, 255), 3)
            
            # aadd poem with large font scale
            if self.current_poem:
                poem_scale = 0.8  # very large scale for poem
                line_spacing = 60  # increased spacing between lines
                start_x = x + w + 70  # position to the right of face
                start_y = y + h - 170 # position a little down (just a tad bit) from the face

                for i, line in enumerate(self.current_poem):
                    y_position = start_y + i * line_spacing
                    self.add_text_with_background(frame, line,
                                                (start_x, y_position), 
                                                poem_scale,
                                                (255, 255, 255), 4)

        return frame

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                processed_frame = self.process_frame(frame)
                cv2.imshow('Emotional Webcam Poetry', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    print("Starting Emotional Webcam Poetry...")
    print("Press 'q' to quit")
    app = EmotionalWebcamPoetry()
    app.run()

if __name__ == "__main__":
    main()