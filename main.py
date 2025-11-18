import cv2
import mediapipe as mp
import numpy as np
import os


# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Window dimensions
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720


# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================


mp_pose = mp.solutions.pose # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def twenty_four_second_violation(landmarks):
    """
    Detect if the user tapping the top of the head with either hand.
    
    This function uses MediaPipe Pose landmarks to determine if the
    user is tapping the top of their head with either hand.
    
    Key landmarks used:
    - Landmark #7: Left ear
    - Landmark #8: Right ear
    - Landmark #17: Left pinky finger
    - Landmark #18: Right pinky finger
    - Landmark #13: Left elbow
    - Landmark #14: Right elbow
    Args:
        landmarks: MediaPipe pose landmarks object containing 33 3D points
        
    Returns:
        bool: True the head tap is detected, False otherwise
    """
    # Get relevant landmarks
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    left_pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]
    right_pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    hand_close_to_head_threshold = 0.3  # Adjust this threshold as needed

    hand_distance_to_nose_left = abs(left_pinky.y - nose.y)
    hand_distance_to_nose_right = abs(right_pinky.y - nose.y)

    # Check if either hand is tapping the head
    if (left_elbow.y < left_ear.y and left_pinky.x < left_ear.x and hand_distance_to_nose_left < hand_close_to_head_threshold
    ) or (right_elbow.y < right_ear.y and  left_ear.x > right_pinky.x < right_ear.x and hand_distance_to_nose_right < hand_close_to_head_threshold):
        return True
    else:
        return False

def traveling_violation(landmarks):
    """
    Detect if the user spinning one arm around another near their chest
    
    
    Key landmarks used:
    - Landmark #7: Left ear
    - Landmark #8: Right ear
    - Landmark #17: Left pinky finger
    - Landmark #18: Right pinky finger
    - Landmark #13: Left elbow
    - Landmark #14: Right elbow
    Args:
        landmarks: MediaPipe pose landmarks object containing 33 3D points
        
    Returns:
        bool: True the travel call is detected, False otherwise
    """
    #Get the landmarks
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # Check if the arms are crossed in front of the chest
    if(left_elbow.x > left_wrist.x > right_elbow.x and right_elbow.x < right_wrist.x < left_elbow.x and
        left_wrist.y > left_shoulder.y and right_wrist.y > right_shoulder.y):
        return True
    else:
        return False


# ============================================================================
# WEB CAMERA SETUP
# ============================================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Image Output', WINDOW_WIDTH, WINDOW_HEIGHT)

print("Starting webcam feed...")
print("Press 'q' to quit.")

# ============================================================================
# MAIN LOOP
# ============================================================================

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "UNKNOWN"
        results_pose = pose.process(image_rgb)

        if results_pose.pose_landmarks:
            # Get pose landmarks
            landmarks = results_pose.pose_landmarks.landmark
            # Check for head tap gesture
            if twenty_four_second_violation(landmarks):
                current_state = "HEAD_TAP"
            elif traveling_violation(landmarks):
                current_state = "TRAVEL CALL"
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        if current_state == "HEAD_TAP":
            cv2.putText(frame, "Head Tap Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if current_state == "TRAVEL CALL":
            cv2.putText(frame, "TRAVEL CALL Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()