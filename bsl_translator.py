"""
BSL (British Sign Language) Translator
Real-time hand gesture recognition for BSL alphabet
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import json
import os

class BSLTranslator:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand detection configuration
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Translation state
        self.current_letter = ""
        self.last_letter = ""
        self.sentence = ""
        self.letter_buffer = deque(maxlen=10)  # Smooth predictions
        self.last_detection_time = time.time()
        self.detection_cooldown = 1.5  # seconds between letter additions
        
        # Load gesture database
        self.gesture_db = self.load_gesture_database()
        
    def load_gesture_database(self):
        """Load or create gesture database for BSL alphabet"""
        db_path = "/home/claude/bsl_gestures.json"
        
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                return json.load(f)
        else:
            # Create basic database with gesture definitions
            return self.create_default_gestures()
    
    def create_default_gestures(self):
        """
        Create default gesture patterns for BSL alphabet
        This uses hand landmark relationships as features
        """
        # Simplified gesture patterns based on finger states and hand orientation
        gestures = {
            "A": {"fingers_extended": [0, 0, 0, 0, 1], "thumb_across_palm": True},
            "B": {"fingers_extended": [0, 1, 1, 1, 1], "palm_orientation": "forward"},
            "C": {"fingers_curved": True, "thumb_gap": "medium"},
            "D": {"fingers_extended": [1, 0, 0, 0, 0], "circle_with_thumb": True},
            "E": {"fingers_curved": True, "all_bent": True},
            "F": {"fingers_extended": [0, 0, 1, 1, 1], "thumb_touch_index": True},
            "G": {"fingers_extended": [1, 0, 0, 0, 0], "horizontal": True},
            "H": {"fingers_extended": [1, 1, 0, 0, 0], "horizontal": True},
            "I": {"fingers_extended": [0, 0, 0, 0, 1], "straight_up": True},
            "J": {"fingers_extended": [0, 0, 0, 0, 1], "hook_motion": True},
            "K": {"fingers_extended": [1, 1, 0, 0, 0], "v_shape": True},
            "L": {"fingers_extended": [1, 0, 0, 0, 1], "perpendicular": True},
            "M": {"fingers_extended": [0, 1, 1, 1, 1], "thumb_under": True},
            "N": {"fingers_extended": [0, 1, 1, 0, 0], "thumb_under": True},
            "O": {"fingers_curved": True, "circle_shape": True},
            "P": {"fingers_extended": [1, 0, 0, 0, 0], "pointing_down": True},
            "Q": {"fingers_extended": [1, 0, 0, 0, 1], "pointing_down": True},
            "R": {"fingers_extended": [0, 1, 1, 0, 0], "crossed": True},
            "S": {"fingers_extended": [0, 0, 0, 0, 0], "fist": True},
            "T": {"fingers_extended": [1, 0, 0, 0, 0], "thumb_between": True},
            "U": {"fingers_extended": [0, 1, 1, 0, 0], "together": True},
            "V": {"fingers_extended": [0, 1, 1, 0, 0], "apart": True},
            "W": {"fingers_extended": [0, 1, 1, 1, 0], "spread": True},
            "X": {"fingers_extended": [1, 0, 0, 0, 0], "bent": True},
            "Y": {"fingers_extended": [0, 0, 0, 0, 1], "thumb_out": True},
            "Z": {"fingers_extended": [1, 0, 0, 0, 0], "z_motion": True}
        }
        return gestures
    
    def add_space(self):
        """Insert a space into the translation and reset state gate."""
        self.sentence += " "
        self.last_letter = ""
        print("Space added")
    
    def clear_translation(self):
        """Clear the current translation."""
        self.sentence = ""
        self.last_letter = ""
        print("Translation cleared")
    
    def remove_last_character(self):
        """Remove the most recent character from the translation."""
        if self.sentence:
            self.sentence = self.sentence[:-1]
            self.last_letter = ""
            print("Removed last character")
    
    def handle_keyboard_input(self, key):
        """
        Centralized keyboard control handler.
        Returns False to signal the main loop to exit.
        """
        NO_KEY = 255
        if key == NO_KEY:
            return True
        
        quit_keys = {ord('q'), ord('Q'), 27}  # q/Q or Esc
        clear_keys = {ord('c'), ord('C')}
        space_keys = {ord(' '), ord('\r'), ord('\n')}
        undo_keys = {8, 127}  # Backspace / Delete
        
        if key in quit_keys:
            return False
        if key in clear_keys:
            self.clear_translation()
        elif key in space_keys:
            self.add_space()
        elif key in undo_keys:
            self.remove_last_character()
        
        return True
    
    def calculate_hand_features(self, hand_landmarks):
        """
        Extract features from hand landmarks for gesture recognition
        Returns a feature dictionary
        """
        landmarks = hand_landmarks.landmark
        
        # Get landmark coordinates
        def get_coords(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        
        # Finger tip and base indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_bases = [2, 5, 9, 13, 17]
        
        # Calculate which fingers are extended
        fingers_extended = []
        for tip, base in zip(finger_tips, finger_bases):
            tip_pos = get_coords(tip)
            base_pos = get_coords(base)
            # Finger is extended if tip is higher/further than base
            is_extended = tip_pos[1] < base_pos[1] - 0.05
            fingers_extended.append(int(is_extended))
        
        # Calculate palm center
        palm_center = np.mean([get_coords(i) for i in [0, 5, 9, 13, 17]], axis=0)
        
        # Calculate hand orientation
        wrist = get_coords(0)
        middle_base = get_coords(9)
        palm_vector = middle_base - wrist
        
        # Distance features
        thumb_index_dist = np.linalg.norm(get_coords(4) - get_coords(8))
        finger_spread = np.linalg.norm(get_coords(8) - get_coords(20))
        
        # Hand openness (average distance from palm center)
        openness = np.mean([
            np.linalg.norm(get_coords(tip) - palm_center) 
            for tip in finger_tips
        ])
        
        features = {
            "fingers_extended": fingers_extended,
            "thumb_index_dist": thumb_index_dist,
            "finger_spread": finger_spread,
            "openness": openness,
            "palm_angle": np.arctan2(palm_vector[1], palm_vector[0]),
            "num_extended": sum(fingers_extended)
        }
        
        return features
    
    def recognize_gesture(self, features):
        """
        Match extracted features to BSL gestures
        Returns recognized letter or None
        """
        fingers = features["fingers_extended"]
        num_extended = features["num_extended"]
        openness = features["openness"]
        thumb_index_dist = features["thumb_index_dist"]
        
        # Simple rule-based recognition (would be ML-based in production)
        # Based on finger extension patterns
        
        # Fist (S)
        if num_extended == 0:
            return "S"
        
        # Only pinky extended (I or Y)
        if fingers == [0, 0, 0, 0, 1]:
            if features["finger_spread"] > 0.3:
                return "Y"
            return "I"
        
        # Thumb and pinky extended (L or Y)
        if fingers == [1, 0, 0, 0, 1]:
            if abs(features["palm_angle"]) < 0.5:
                return "L"
            return "Y"
        
        # Index extended (D, G, or pointing)
        if fingers == [1, 0, 0, 0, 0]:
            if openness < 0.15:
                return "D"
            return "G"
        
        # Index and middle (U, V, R, K, H)
        if fingers == [0, 1, 1, 0, 0]:
            if features["finger_spread"] < 0.1:
                return "U"
            elif features["finger_spread"] > 0.15:
                return "V"
            return "R"
        
        if fingers == [1, 1, 0, 0, 0]:
            return "K"
        
        # Three fingers extended (W)
        if fingers == [0, 1, 1, 1, 0]:
            return "W"
        
        # All fingers extended (B)
        if num_extended == 4 and fingers[0] == 0:
            return "B"
        
        # All including thumb (5)
        if num_extended == 5:
            return "5"
        
        # Open hand but curved (C or O)
        if openness > 0.2 and num_extended <= 2:
            if thumb_index_dist < 0.15:
                return "O"
            return "C"
        
        # Thumb across palm (A)
        if fingers == [0, 0, 0, 0, 1] and thumb_index_dist < 0.1:
            return "A"
        
        # Default for unrecognized
        return None
    
    def process_frame(self, frame):
        """Process a single frame for hand detection and gesture recognition"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and recognize gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract features and recognize gesture
                features = self.calculate_hand_features(hand_landmarks)
                detected_letter = self.recognize_gesture(features)
                
                if detected_letter:
                    self.letter_buffer.append(detected_letter)
                    
                    # Get most common letter in buffer (smoothing)
                    if len(self.letter_buffer) >= 5:
                        from collections import Counter
                        most_common = Counter(self.letter_buffer).most_common(1)[0][0]
                        self.current_letter = most_common
        else:
            self.current_letter = ""
        
        # Add letter to sentence if stable and cooldown passed
        current_time = time.time()
        if (self.current_letter and 
            self.current_letter != self.last_letter and
            current_time - self.last_detection_time > self.detection_cooldown):
            
            if self.current_letter == "SPACE":
                self.sentence += " "
            else:
                self.sentence += self.current_letter
            
            self.last_letter = self.current_letter
            self.last_detection_time = current_time
        
        return frame
    
    def draw_ui(self, frame):
        """Draw UI elements on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for UI
        overlay = frame.copy()
        
        # Top bar for current detection
        cv2.rectangle(overlay, (0, 0), (width, 80), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Current letter (large)
        if self.current_letter:
            cv2.putText(frame, self.current_letter, (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, "Detected", (120, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Show BSL sign", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Bottom bar for sentence
        cv2.rectangle(frame, (0, height - 100), (width, height), (50, 50, 50), -1)
        
        # Sentence display
        cv2.putText(frame, "Translation:", (20, height - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Word wrap the sentence
        sentence_display = self.sentence[-50:] if len(self.sentence) > 50 else self.sentence
        cv2.putText(frame, sentence_display, (20, height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions
        instruction_text = "Controls: C=clear | Space/Enter=space | Backspace=undo | Q/Esc=quit"
        instruction_x = max(20, width - 650)
        cv2.putText(frame, instruction_text, 
                   (instruction_x, height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop for the BSL translator"""
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("BSL Translator Started!")
        print("Controls:")
        print("  C or c - Clear translation")
        print("  Space / Enter - Add space")
        print("  Backspace / Delete - Remove last character")
        print("  Q, q, or Esc - Quit")
        print("\nSupported letters: A-Z (subset implemented)")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            # Display
            cv2.imshow('BSL Translator', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_keyboard_input(key):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Print final translation
        print("\n" + "="*50)
        print("Final Translation:")
        print(self.sentence)
        print("="*50)

def main():
    translator = BSLTranslator()
    translator.run()

if __name__ == "__main__":
    main()
