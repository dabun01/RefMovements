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

    left_head_tap = (
        left_elbow.y < left_ear.y and 
        left_pinky.x < left_ear.x and 
        hand_distance_to_nose_left < hand_close_to_head_threshold
    )

    right_head_tap = (
        right_elbow.y < right_ear.y and 
        right_pinky.x > right_ear.x and 
        hand_distance_to_nose_right < hand_close_to_head_threshold
    )

    # Check if either hand is tapping the head
    if left_head_tap or right_head_tap:
        return True
    else:
        return False

# Global variables to store previous wrist positions
prev_left_wrist = None
prev_right_wrist = None

def traveling_violation(landmarks):
    global prev_left_wrist, prev_right_wrist
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

    arms_crossed = (
        left_elbow.x > left_wrist.x > right_elbow.x and
        right_elbow.x < right_wrist.x < left_elbow.x and
        left_wrist.y > left_shoulder.y and right_wrist.y > right_shoulder.y
    )

    if prev_left_wrist is None or prev_right_wrist is None:
        # Initialize previous wrist positions
        prev_left_wrist = left_wrist
        prev_right_wrist = right_wrist
        return False
    
    swapping = (
        (prev_left_wrist.x < prev_right_wrist.x and left_wrist.x > right_wrist.x) or
        (prev_left_wrist.x > prev_right_wrist.x and left_wrist.x < right_wrist.x)
    )

    # Update previous wrist positions
    prev_left_wrist = left_wrist
    prev_right_wrist = right_wrist

    if swapping and arms_crossed:
        return True
    else:
        return False
    

prev_left_index = None
prev_right_index = None

def carrying_violation(landmarks):
    global prev_left_index, prev_right_index
    """
    Detect if the user is swapping their palm up and dow with outstretched arm.
    
    Key landmarks used:
    - Landmark #15: Left wrist
    - Landmark #16: Right wrist
    - Landmark #13: Left elbow
    - Landmark #14: Right elbow
    Args:
        landmarks: MediaPipe pose landmarks object containing 33 3D points
        
    Returns:
        bool: True the carrying call is detected, False otherwise
    """

    # Get relevant landmarks
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]

    if prev_left_index is None or prev_right_index is None:
        # Initialize previous wrist positions
        prev_left_index = left_index
        prev_right_index = right_index
        return False
    
    index_swapping_left = (
        (prev_left_index.y > left_index.y) or (prev_left_index.y < left_index.y)
    )

    index_swapping_right = (
        (prev_right_index.y > right_index.y) or (prev_right_index.y < right_index.y)
    )

    # Update previous wrist positions
    prev_left_index = left_index
    prev_right_index = right_index

    left_arm_straight = (left_wrist.x > left_elbow.x and left_elbow.x > left_shoulder.x)
    right_arm_straight = (right_wrist.x < right_elbow.x and right_elbow.x < right_shoulder.x)

    hand_distance_threshold = 0.1  # Adjust this threshold as needed
    hand_distance_elbow_left = abs(left_wrist.y - left_elbow.y)
    hand_distance_elbow_right = abs(right_wrist.y - right_elbow.y)

    hand_far_from_elbow_left = (hand_distance_elbow_left < hand_distance_threshold)
    hand_far_from_elbow_right = (hand_distance_elbow_right < hand_distance_threshold)

    left_arm_carrying = (left_arm_straight and hand_far_from_elbow_right and index_swapping_left)
    right_arm_carrying = (right_arm_straight and hand_far_from_elbow_left and index_swapping_right)

    if left_arm_carrying or right_arm_carrying:
        return True
    else:
        return False
    

def technical_foul_violation(landmarks):
    """
    Placeholder for technical foul detection logic.
    
    Args:
        landmarks: MediaPipe pose landmarks object containing 33 3D points
    Returns:
        bool: True if technical foul is detected, False otherwise
    """

    #Get relevant landmarks
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]
    right_pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]
    left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
    left_thumb = landmarks[mp_pose.PoseLandmark.LEFT_THUMB]
    right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB]

    fingers_up_left_hand_threshold = 0.1  # Adjust this threshold as needed

    fingers_up_left_hand = (
        left_index.y < left_wrist.y and
        left_pinky.y < left_wrist.y and
        left_thumb.y < left_wrist.y
    )

    fingers_up_right_hand = (
        right_index.y < right_wrist.y and
        right_pinky.y < right_wrist.y and
        right_thumb.y < right_wrist.y
    )

    t_signal_left = (
        left_elbow.x > left_wrist.x > right_elbow.x and
        right_elbow.x < right_wrist.x < left_elbow.x and
        left_elbow.y > right_elbow.y and
        right_wrist.y < left_elbow.y and
        left_index.y > right_index.y
    )

    t_signal_right = (
        right_elbow.x < right_wrist.x < left_elbow.x and
        left_elbow.x > left_wrist.x > right_elbow.x and
        right_elbow.y > left_elbow.y and
        left_wrist.y < right_elbow.y
    )

    hands_close_together_threshold = 0.2  # Adjust this threshold as needed
    hands_distance_x = abs(left_index.x - right_index.x)
    hands_distance_y = abs(left_index.y - right_index.y)
    hands_close_together = hands_distance_x < hands_close_together_threshold and hands_distance_y < hands_close_together_threshold

    if (fingers_up_left_hand > fingers_up_left_hand_threshold and t_signal_left and hands_close_together):
        return True
    else:
        return False

def blocking_violation(landmarks):
    """
    Placeholder for blocking violation detection logic.
    
    Args:
        landmarks: MediaPipe pose landmarks object containing 33 3D points
    Returns:
        bool: True if blocking violation is detected, False otherwise
    """
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    arms_at_hip_level = (
        abs(left_wrist.y - left_hip.y) < 0.1 and
        abs(right_wrist.y - right_hip.y) < 0.1
    )
    if arms_at_hip_level:
        return True
    else:
        return False

# ============================================================================
# Prepare the images
# ============================================================================
print("=" * 60)
print("Basketball Referee Call Detection System")
print("=" * 60)

#Load images from the images folder
travel_img = cv2.imread(os.path.join('images', 'travel.png'))
carry_img = cv2.imread(os.path.join('images', 'carrying.jpg'))
tech_foul_img = cv2.imread(os.path.join('images', 'technicalFoul.jpg'))
head_tap_img = cv2.imread(os.path.join('images', 'headTap.png'))
blocking_img = cv2.imread(os.path.join('images', 'blocking.jpg'))

#Verify images loaded correctly
if travel_img is None:
    print("Error: Could not load travel.png")
    exit()
if carry_img is None:
    print("Error: Could not load carrying.jpg")
    exit()
if tech_foul_img is None:
    print("Error: Could not load technicalFoul.jpg")
    exit()
if head_tap_img is None:
    print("Error: Could not load headTap.png")
    exit()
if blocking_img is None:
    print("Error: Could not load blocking.jpg")
    exit()

# Resize images to fit window
travel_img = cv2.resize(travel_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
carry_img = cv2.resize(carry_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
tech_foul_img = cv2.resize(tech_foul_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
head_tap_img = cv2.resize(head_tap_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
blocking_img = cv2.resize(blocking_img, (WINDOW_WIDTH, WINDOW_HEIGHT))

# ============================================================================
# WEB CAMERA SETUP
# ============================================================================
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Image Output', WINDOW_WIDTH, WINDOW_HEIGHT)

current_image = head_tap_img.copy()

print("Starting webcam feed...")
print("Press 'q' to quit.")

# ============================================================================
# MAIN LOOP
# ============================================================================
draw_landmarks = False
current_video = None   
video_playing = False

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

            if twenty_four_second_violation(landmarks):
                current_image = head_tap_img.copy()
                cv2.putText(frame, "Head Tap Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif traveling_violation(landmarks):
                current_image = travel_img.copy()
                cv2.putText(frame, "TRAVEL CALL Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif carrying_violation(landmarks):
                current_image = carry_img.copy()
                cv2.putText(frame, "CARRYING CALL Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif blocking_violation(landmarks):
                current_image = blocking_img.copy()
                cv2.putText(frame, "BLOCKING CALL Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif technical_foul_violation(landmarks):
                current_video = cv2.VideoCapture(os.path.join('images', 'shaqTech.mp4'))
                video_playing = True
                cv2.putText(frame, "TECHNICAL FOUL Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw pose landmarks
            if cv2.waitKey(2) & 0xFF == ord('s'):
                draw_landmarks = not draw_landmarks

            if draw_landmarks == True:
                mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

        cv2.imshow('Camera Feed', frame)

        if video_playing and current_video is not None:
            ret, vid_frame = current_video.read()
            if ret:
                vid_frame = cv2.resize(vid_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                cv2.imshow('Image Output', vid_frame)
            else:
                # End of video → stop playback and clear video
                current_video.release()
                current_video = None
                video_playing = False
        else:
            # show default image (static)
            cv2.imshow('Image Output', current_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()