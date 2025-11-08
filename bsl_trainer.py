"""
BSL Gesture Trainer
Tool for collecting and training custom BSL gestures
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

class BSLGestureTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.training_data = {}
        self.current_letter = None
        self.samples_collected = 0
        self.samples_per_letter = 30
        
    def extract_landmark_features(self, hand_landmarks):
        """Extract normalized landmark positions"""
        landmarks = hand_landmarks.landmark
        
        # Get all landmark coordinates
        coords = []
        for lm in landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        
        # Normalize to wrist position
        wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
        normalized = []
        for i in range(0, len(coords), 3):
            normalized.extend([
                coords[i] - wrist_x,
                coords[i+1] - wrist_y,
                coords[i+2] - wrist_z
            ])
        
        return normalized
    
    def collect_samples(self):
        """Collect training samples for each letter"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letter_index = 0
        collecting = False
        countdown = 0
        
        print("BSL Gesture Trainer")
        print("=" * 50)
        print("Instructions:")
        print("1. Press SPACE or ENTER to start collecting samples for a letter")
        print("2. Hold the sign steady for 3 seconds")
        print("3. Move to next letter automatically")
        print("4. Press 'Q' or Esc to quit and save")
        print("=" * 50)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            height, width = frame.shape[:2]
            
            # Draw UI
            if letter_index < len(alphabet):
                current_letter = alphabet[letter_index]
                
                # Instructions box
                cv2.rectangle(frame, (20, 20), (width - 20, 150), (50, 50, 50), -1)
                cv2.putText(frame, f"Current Letter: {current_letter}", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(frame, f"Samples: {self.samples_collected}/{self.samples_per_letter}", 
                           (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if not collecting:
                    cv2.putText(frame, "Press SPACE/ENTER to start collecting", (40, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                else:
                    cv2.putText(frame, f"Hold steady... {countdown}", (40, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Progress bar
                progress = (letter_index / len(alphabet)) * 100
                cv2.rectangle(frame, (40, height - 50), (width - 40, height - 30), 
                             (100, 100, 100), -1)
                cv2.rectangle(frame, (40, height - 50), 
                             (int(40 + (width - 80) * progress / 100), height - 30), 
                             (0, 255, 0), -1)
                cv2.putText(frame, f"Progress: {letter_index}/{len(alphabet)} letters", 
                           (40, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Training Complete! Press Q or Esc to save and exit", 
                           (width//2 - 300, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Collect sample if in collection mode
                    if collecting and letter_index < len(alphabet):
                        features = self.extract_landmark_features(hand_landmarks)
                        current_letter = alphabet[letter_index]
                        
                        if current_letter not in self.training_data:
                            self.training_data[current_letter] = []
                        
                        self.training_data[current_letter].append(features)
                        self.samples_collected += 1
                        countdown = self.samples_collected
                        
                        # Check if enough samples collected
                        if self.samples_collected >= self.samples_per_letter:
                            print(f"Completed letter {current_letter}")
                            letter_index += 1
                            self.samples_collected = 0
                            collecting = False
            
            cv2.imshow('BSL Gesture Trainer', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord(' '), ord('\r'), ord('\n')) and not collecting and letter_index < len(alphabet):
                collecting = True
                countdown = 0
                print(f"Collecting samples for {alphabet[letter_index]}...")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Save training data
        self.save_training_data()
    
    def save_training_data(self):
        """Save collected training data to JSON file"""
        if not self.training_data:
            print("No training data to save")
            return
        
        filename = f"bsl_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"/home/claude/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        
        print(f"\nTraining data saved to: {filepath}")
        print(f"Total letters trained: {len(self.training_data)}")
        print(f"Total samples: {sum(len(samples) for samples in self.training_data.values())}")
        
        return filepath

def main():
    trainer = BSLGestureTrainer()
    trainer.collect_samples()

if __name__ == "__main__":
    main()
