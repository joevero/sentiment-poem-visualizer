import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import time
from typing import List, Tuple
from fer import FER  # Facial Emotion Recognition
import emoji
import tensorflow

class EmotionalPoemGenerator:
    def __init__(self):
        # Happy/motivational poems
        self.happy_poems = [
            
            [
                "Sunshine in your smile",
                "Keep that joy burning so bright",
                "Light up the whole world"
            ],
            [
                "Your happiness glows",
                "Like stars dancing in the night",
                "Never let it fade"
            ],
            [
                "Joy flows from within",
                "A beacon of hope and light",
                "Stay radiant now"
            ]
        ]

        # Uplifting poems for sad moments
        self.uplifting_poems = [
            [
                "After the rainfall",
                "Rainbow colors paint the sky",
                "Hope blooms anew now"
            ],
            [
                "Dark clouds will pass by",
                "Sun always breaks through the grey",
                "Your strength guides the way"
            ],
            [
                "Each tear that falls down",
                "Waters seeds of tomorrow",
                "New joy will spring up"
            ]
        ]

        # Sylvia Plath inspired poems for neutral
        self.neutral_poems = [
            [
                "Mirror reflects truth",
                "Silver and exact, unmoved",
                "Time passes, face fades"
            ],
            [
                "Between light and dark",
                "The moon's face tells no stories",
                "Silence speaks volumes"
            ],
            [
                "Thoughts spiral like moths",
                "Around the flame of being",
                "Neither here nor there"
            ]
        ]

    def get_poem(self, emotion: str) -> Tuple[List[str], str, str]:
        """Get a poem based on emotional state"""
        if emotion == "happy":
            return (
                random.choice(self.happy_poems),
                "You're feeling happy! Keep shining! ☀️",
                "😊"
            )
        elif emotion == "sad":
            return (
                random.choice(self.uplifting_poems),
                "Let's cheer you up! 🌈",
                "😢"
            )
        else:  # neutral
            return (
                random.choice(self.neutral_poems),
                "You seem neutral. Change how you feel with this Sylvia Plath poem 🎭",
                "😐"
            )

class EmotionalWebcamPoetry:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # start face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # iinitialize emotion detector
        self.emotion_detector = FER(mtcnn=True)
        
        # start that poem generator
        self.poem_generator = EmotionalPoemGenerator()
        
        # Store current poem and its timestamp
        self.current_poem = None
        self.current_message = ""
        self.current_emoji = ""
        self.poem_timestamp = time.time()
        self.poem_refresh_interval = 3  # check emotions every 3 seconds

    def add_text_to_image(self, 
                         image: np.ndarray, 
                         text: List[str], 
                         message: str,
                         emoji_text: str,
                         position: Tuple[int, int], 
                         font_size: int = 20,
                         color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Add multiline text and emotion message to image"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Draw emotion message and emoji
        x, y = position
        draw.text((x, y - 30), f"{message} {emoji_text}", font=font, fill=color)
        
        # Draw poem
        line_height = int(font_size * 1.5)  # Increase line height to take up more space
        for i, line in enumerate(text):
            draw.text((x, y + i * line_height), line, font=font, fill=color)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def detect_emotion(self, frame: np.ndarray) -> str:
        """Detect emotion in frame"""
        emotions = self.emotion_detector.detect_emotions(frame)
        
        if emotions:
            # get the DOMINANT emotion
            emotions_dict = emotions[0]['emotions']
            if emotions_dict['happy'] > 0.4:
                return "happy"
            elif emotions_dict['sad'] > 0.3:
                return "sad"
            else:
                return "neutral"
        
        return "neutral"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Check if we need to update emotion and poem
        current_time = time.time()
        if current_time - self.poem_timestamp > self.poem_refresh_interval:
            if len(faces) > 0:
                emotion = self.detect_emotion(frame)
                self.current_poem, self.current_message, self.current_emoji = (
                    self.poem_generator.get_poem(emotion)
                )
            self.poem_timestamp = current_time

        # process each detected face
        for (x, y, w, h) in faces:
            # draw box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # add poem and emotion message next to the face
            if self.current_poem:
                frame = self.add_text_to_image(
                    frame,
                    self.current_poem,
                    self.current_message,
                    self.current_emoji,
                    (x + w + 10, y),
                    font_size=200,
                    color=(255, 255, 255)
                )

        return frame

    def run(self):
        """Main loop for webcam processing"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # pprocess the frame
                processed_frame = self.process_frame(frame)

                # display the result
                cv2.imshow('Emotional Webcam Poetry', processed_frame)

                # break loop on 'q' press, make sure in python window, not IDE
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    print("Starting Emotional Webcam Poetry...")
    print("Press 'q' to quit")
    print("The program will analyze your emotions and generate appropriate poems...")
    app = EmotionalWebcamPoetry()
    app.run()

if __name__ == "__main__":
    main()