import cv2
import mediapipe as mp
import time
import math  # For distance and angle calculations

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Focused on single hand recognition for words
    min_detection_confidence=0.5, # Adjusted confidence
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils # Corrected initialization

# Finger tip and joint landmarks
# (TIP, IP/DIP, PIP, MCP, BASE) where BASE is the wrist
# Note: Thumb only has IP joint between TIP and MCP.
FINGER_LANDMARKS_EXTENDED = {
    "thumb": (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_MCP,
              mp_hands.HandLandmark.WRIST),
    "index": (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP,
              mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP,
              mp_hands.HandLandmark.WRIST),
    "middle": (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
               mp_hands.HandLandmark.WRIST),
    "ring": (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP,
             mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.WRIST),
    "pinky": (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_PIP,
              mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.WRIST),
}


# Helper function to get 3D coordinates of a landmark
def get_landmark_coords(landmarks, landmark_enum):
    """Returns [x, y, z] coordinates for a given landmark."""
    return [
        landmarks.landmark[landmark_enum].x,
        landmarks.landmark[landmark_enum].y,
        landmarks.landmark[landmark_enum].z
    ]


# Helper function to calculate Euclidean distance between two 3D points
def get_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# Helper function to calculate angle between three points (A-B-C)
# Returns angle in degrees at point B
def get_angle(p1, p2, p3):
    """Calculates the angle (in degrees) at point P2 formed by vectors P2P1 and P2P3."""
    v1 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]]

    dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0  # Avoid division by zero

    cosine_angle = min(max(dot_product / (magnitude_v1 * magnitude_v2), -1), 1)
    angle_rad = math.acos(cosine_angle)
    return math.degrees(angle_rad)


# REFINED: Check if a finger is extended (straight) based ONLY on joint angles
def is_finger_extended(landmarks, finger_name, angle_threshold=30): # Adjusted angle threshold
    """Checks if a given finger is extended (straight) based on joint angles."""
    finger_landmarks_tuple = FINGER_LANDMARKS_EXTENDED[finger_name]

    if finger_name == "thumb":
        tip_lm, ip_lm, mcp_lm, _ = finger_landmarks_tuple
        mcp_coords = get_landmark_coords(landmarks, mcp_lm)
        ip_coords = get_landmark_coords(landmarks, ip_lm)
        tip_coords = get_landmark_coords(landmarks, tip_lm)
        angle = get_angle(mcp_coords, ip_coords, tip_coords)
        return angle < angle_threshold
    else:
        tip_lm, dip_lm, pip_lm, mcp_lm, _ = finger_landmarks_tuple
        tip_coords = get_landmark_coords(landmarks, tip_lm)
        dip_coords = get_landmark_coords(landmarks, dip_lm)
        pip_coords = get_landmark_coords(landmarks, pip_lm)
        mcp_coords = get_landmark_coords(landmarks, mcp_lm)

        angle_dip = get_angle(tip_coords, dip_coords, pip_coords)
        angle_pip = get_angle(dip_coords, pip_coords, mcp_coords)
        return angle_dip < angle_threshold and angle_pip < angle_threshold


# REFINED: Check if a finger is curled (bent) based ONLY on joint angles
def is_finger_curled(landmarks, finger_name, angle_threshold=60): # Significantly lowered threshold for curl
    """Checks if a given finger is curled (bent) based on joint angles."""
    finger_landmarks_tuple = FINGER_LANDMARKS_EXTENDED[finger_name]

    if finger_name == "thumb":
        tip_lm, ip_lm, mcp_lm, _ = finger_landmarks_tuple
        mcp_coords = get_landmark_coords(landmarks, mcp_lm)
        ip_coords = get_landmark_coords(landmarks, ip_lm)
        tip_coords = get_landmark_coords(landmarks, tip_lm)
        angle = get_angle(mcp_coords, ip_coords, tip_coords)
        return angle > angle_threshold
    else:
        tip_lm, dip_lm, pip_lm, mcp_lm, _ = finger_landmarks_tuple
        tip_coords = get_landmark_coords(landmarks, tip_lm)
        dip_coords = get_landmark_coords(landmarks, dip_lm)
        pip_coords = get_landmark_coords(landmarks, pip_lm)
        mcp_coords = get_landmark_coords(landmarks, mcp_lm)

        angle_dip = get_angle(tip_coords, dip_coords, pip_coords)
        angle_pip = get_angle(dip_coords, pip_coords, mcp_coords)
        return angle_dip > angle_threshold or angle_pip > angle_threshold


# Main recognition function for words
def recognize_word_gesture(landmarks):
    """Recognizes specific word gestures based on finger states and positions."""

    # Get finger extension/curl states using the refined functions
    thumb_ext = is_finger_extended(landmarks, "thumb")
    index_ext = is_finger_extended(landmarks, "index")
    middle_ext = is_finger_extended(landmarks, "middle")
    ring_ext = is_finger_extended(landmarks, "ring")
    pinky_ext = is_finger_extended(landmarks, "pinky")

    thumb_curled = is_finger_curled(landmarks, "thumb")
    index_curled = is_finger_curled(landmarks, "index")
    middle_curled = is_finger_curled(landmarks, "middle")
    ring_curled = is_finger_curled(landmarks, "ring")
    pinky_curled = is_finger_curled(landmarks, "pinky")

    # Get landmark coordinates for proximity checks
    thumb_tip = get_landmark_coords(landmarks, mp_hands.HandLandmark.THUMB_TIP)
    index_tip = get_landmark_coords(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)

    wrist = get_landmark_coords(landmarks, mp_hands.HandLandmark.WRIST)

    # Calculate relative distance thresholds for robustness across different hand sizes/camera distances
    wrist_pinky_mcp_dist = get_distance(wrist, get_landmark_coords(landmarks, mp_hands.HandLandmark.PINKY_MCP))

    # Thresholds for 'touching' or 'close' (as a fraction of hand size)
    touch_threshold_strict = wrist_pinky_mcp_dist * 0.15
    touch_threshold_loose = wrist_pinky_mcp_dist * 0.25

    # --- Word Gestures ---
    # These are static approximations for dynamic signs.
    # The "OK" sign is a distinct static gesture.
    # For "Hi", "Hello", "Thank You", "Bye", a simple "Open Hand" is used as a static representation.
    # True recognition for these words requires tracking hand movement over time.

    # 1. OK Sign (similar to ASL 'F')
    # Thumb and index tips touching, others extended (straight).
    # NOTE: Because thumb and index tips are touching, their joints will be bent, so they will register as 'curled'.
    if thumb_curled and index_curled and middle_ext and ring_ext and pinky_ext:
        if get_distance(thumb_tip, index_tip) < touch_threshold_strict:
            return "OK"

    # 2. OPEN HAND (Approximation for Hi/Hello/Thank You/Bye)
    # All fingers extended (straight) and reasonably spread.
    if thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
        if get_distance(thumb_tip, index_tip) > touch_threshold_loose and \
                get_distance(index_tip, pinky_tip) > touch_threshold_loose: # Ensure reasonable spread
            return "OPEN HAND (Hi/Hello/Thank You/Bye)"

    # 3. FIST (Can be an initial/final pose for some gestures, or signify "Stop")
    # All fingers curled (bent).
    if thumb_curled and index_curled and middle_curled and ring_curled and pinky_curled:
        # Additional check: thumb tip is close to the palm/wrist to confirm tucking
        if get_distance(thumb_tip, wrist) < touch_threshold_loose:
            return "FIST (Possible Bye/Stop)"


    return "N/A"  # No recognized word gesture


# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open video stream. Check camera connection or index.")
    exit()

print("Camera opened successfully.")
print("--- Word Gesture Recognition ---")
print("1. 'OK' Sign: Thumb and index finger tips touching, others extended.")
print("2. 'OPEN HAND': All fingers extended and spread (static approximation for Hi/Hello/Thank You/Bye).")
print("3. 'FIST': All fingers curled (static approximation for Bye/Stop).")
print("\nIMPORTANT: 'Hi', 'Hello', 'Thank You', 'Bye' are typically DYNAMIC gestures and cannot be perfectly recognized by this STATIC code.")
print("The 'OPEN HAND' and 'FIST' are just common static hand shapes that *can* be part of these dynamic words.")
print("Press 'q' to quit.")

recognized_gesture = "N/A"
last_recognition_time = time.time()
display_duration = 1.5  # How long to display a recognized gesture

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    current_gesture_this_frame = "N/A" # Variable for current frame's recognition

    if results.multi_hand_landmarks:
        # Since max_num_hands is 1, we expect at most one hand
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        current_gesture_this_frame = recognize_word_gesture(hand_landmarks)

    # Update the recognized gesture for display
    if current_gesture_this_frame != "N/A":
        # Only update if the gesture has changed or if it's a new recognition
        if current_gesture_this_frame != recognized_gesture:
            recognized_gesture = current_gesture_this_frame
            last_recognition_time = time.time()
    else:  # No gesture is currently detected in this frame
        # If the display duration has passed since the last recognition, clear it
        if (time.time() - last_recognition_time > display_duration):
            recognized_gesture = "N/A"

    # Display the recognized gesture on the frame
    if recognized_gesture != "N/A":  # Only display if recognized_gesture is not N/A
        text_to_display = f"Recognized: {recognized_gesture}"
    else:
        text_to_display = "Show a word gesture (OK, Open Hand, Fist)"

    cv2.putText(frame, text_to_display, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Word Recognizer', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
hands.close()
cap.release()
cv2.destroyAllWindows()