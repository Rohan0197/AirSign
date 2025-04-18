import os
import time
import warnings
from datetime import datetime

# Set the environment variable to suppress Tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
from cv2.typing import MatLike
import tkinter as tk
from tkinter import simpledialog

# Ignore warnings
warnings.filterwarnings("ignore")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize drawing utilities
drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Get the first frame to define the canvas dimensions
_, frame = cap.read()
frame_height, frame_width, _ = frame.shape

# Define an empty canvas
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Control when to process the handtracking
frame_queue = []
real_time = False

# Wait for Tensorflow to load and clear the console
print("Waiting for Tensorflow to load")
time.sleep(1)
os.system("cls" if os.name == "nt" else "clear")

# Ask user if they want to process in real time
if input("Process in Real Time? (y/n):").lower() == "y":
    real_time = True
    print("Processing in Real Time")
    # Keyboard controls
    print("Keyboard Controls:")
    print("Press 'q' to Quit")
    print("Press 's' to save current Canvas")
    print("Press 'n' to save new Signature")
    print("Press 'c' to clear current Canvas")
else:
    print("Processing in Post Processing Mode")
    print("Press 'q' to stop recording")

# Initialize hand detection
hand_det = mp.solutions.hands.Hands(
    static_image_mode=False,  # Set to True if hand isn't moving much
    max_num_hands=1,  # Reduce to 1 hand to prevent false detections
    min_detection_confidence=0.8,  # Increase threshold to detect hand more confidently
    min_tracking_confidence=0.8,  # Ensure stable tracking
)

# Store the previous position of the index finger (initialize as None)
prev_x, prev_y = None, None

# Define Region of Interest (ROI)
ROI_TOP = 100
ROI_BOTTOM = 400
ROI_LEFT = 150
ROI_RIGHT = 500

def process_frame(frame: MatLike, canvas: MatLike):
    global prev_x, prev_y  # Keep track of previous index finger position

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    op = hand_det.process(rgb_frame)
    hands = op.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw landmarks and connections on the frame
            drawing_utils.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Get index finger tip coordinates
            index_finger_tip = hand.landmark[8]
            x = int(index_finger_tip.x * frame_width)
            y = int(index_finger_tip.y * frame_height)

            # Only consider points inside the ROI
            if ROI_LEFT < x < ROI_RIGHT and ROI_TOP < y < ROI_BOTTOM:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 5)

                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None  # Reset if outside the ROI
    else:
        # Reset previous position if no hand is detected
        prev_x, prev_y = None, None

    return frame, canvas

def handle_keyboard_input(key, canvas, real_time):
    if key == ord("q"):
        print("Quitting")
        return True, canvas  # Ensure canvas is returned

    if key == ord("s"):
        now = datetime.now()
        name = f"output_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
        os.makedirs("./output", exist_ok=True)  # Ensure directory exists
        print(f"Canvas saved as {name}")
        cv2.imwrite(f"./output/{name}", canvas)

    if key == ord("n"):
        # Open a Tkinter window to get the username
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        user_name = simpledialog.askstring("Save Signature", "Enter your name:")

        if user_name:
            user_name = user_name.strip().replace(" ", "_")  # Remove spaces for folder name safety
            now = datetime.now()
            signature_folder = f"./signatures/{user_name}"
            os.makedirs(signature_folder, exist_ok=True)  # Ensure folder exists
            signature_path = f"{signature_folder}/signature_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            print(f"Signature saved as {signature_path}")
            cv2.imwrite(signature_path, canvas)
            # Clear the canvas
            canvas[:] = 0

    if key == ord("c") and real_time:
        print("Canvas cleared")
        canvas[:] = 0  # Properly clear the canvas instead of reassigning

    return False, canvas


while True:
    # Read frame from video capture
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if real_time:
        # Process the frame for hand detection (only in real-time mode)
        frame, canvas = process_frame(frame, canvas)
        # Overlay the canvas on the frame
        frame = cv2.add(frame, canvas)
        # Display the canvas
        cv2.imshow("Canvas", canvas)
    else:
        # Save the frame to the queue (only in post-processing mode)
        frame_queue.append(frame)

    cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255, 0, 0), 2)

    # Display the output frame
    cv2.imshow("Output", frame)

    # Keyboard controls and delay between getting next frame
    key = cv2.waitKey(1)
    quit_flag, canvas = handle_keyboard_input(key, canvas, real_time)
    if quit_flag:
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# If not real-time, process the frames in the queue
if not real_time:
    # Clear the console
    os.system("cls" if os.name == "nt" else "clear")
    
    # Inform the user
    remaining = len(frame_queue)
    print(f"Processing the frames ({remaining} frames)")
    
    # Process the frames
    while frame_queue:
        remaining = len(frame_queue)
        if remaining % 100 == 0:
            print(f"Remaining Frames: {remaining}")
        _, canvas = process_frame(frame_queue.pop(0), canvas)
    cv2.imshow("Canvas", canvas)
    
    # Inform the user and wait
    print("Processing complete")
    print("Press any key to exit")
    cv2.waitKey(0)

os.makedirs("./output", exist_ok=True)

# Save the final canvas
now = datetime.now()
name = f"final_output_{now.isoformat().replace(':', '-').replace('.', '-')}.png"
print(f"Final Canvas saved as {name}")
cv2.imwrite(f"./output/{name}", canvas)

# Destroy all windows
cv2.destroyAllWindows()
