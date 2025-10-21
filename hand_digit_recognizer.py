import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # We'll focus on recognizing digits from a single hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Finger tip and PIP (Proximal InterPhalangeal) joint landmarks for all fingers
# PIP is the second joint from the tip, MCP is the knuckle.
# We'll use PIP to determine if a finger is extended.
FINGER_LANDMARKS = {
    "thumb": (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_MCP),
    "index": (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
              mp_hands.HandLandmark.INDEX_FINGER_MCP),
    "middle": (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
    "ring": (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP,
             mp_hands.HandLandmark.RING_FINGER_MCP),
    "pinky": (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP),
}


# Function to check if a finger is extended (up)
def is_finger_up(landmarks, finger_name, hand_label):
    tip_lm, pip_lm, mcp_lm = FINGER_LANDMARKS[finger_name]

    # For thumb, check horizontal position relative to MCP for better accuracy
    # Thumb can bend horizontally, so simple Y-axis check isn't enough
    if finger_name == "thumb":
        if hand_label == 'Left':
            # For left hand, thumb tip should be to the right of MCP
            return landmarks.landmark[tip_lm].x > landmarks.landmark[mcp_lm].x and \
                landmarks.landmark[tip_lm].y < landmarks.landmark[pip_lm].y
        else:  # Right hand
            # For right hand, thumb tip should be to the left of MCP
            return landmarks.landmark[tip_lm].x < landmarks.landmark[mcp_lm].x and \
                landmarks.landmark[tip_lm].y < landmarks.landmark[pip_lm].y
    else:
        # For other fingers, check if tip is significantly above the PIP joint
        # This handles cases where fingers are bent but not fully closed
        return landmarks.landmark[tip_lm].y < landmarks.landmark[pip_lm].y


# Function to recognize the digit based on finger states
def recognize_digit(landmarks, hand_label):
    # Determine which fingers are up
    thumb_up = is_finger_up(landmarks, "thumb", hand_label)
    index_up = is_finger_up(landmarks, "index", hand_label)
    middle_up = is_finger_up(landmarks, "middle", hand_label)
    ring_up = is_finger_up(landmarks, "ring", hand_label)
    pinky_up = is_finger_up(landmarks, "pinky", hand_label)

    fingers_up_count = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

    # Simple rule-based recognition for 0-5
    # These rules are heuristic and might need fine-tuning for different hand poses
    if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return 0  # All fingers down
    elif not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return 1  # Only index finger up
    elif not thumb_up and index_up and middle_up and not ring_up and not pinky_up:
        return 2  # Index and Middle up
    elif thumb_up and index_up and middle_up and not ring_up and not pinky_up:
        return 3  # Thumb, Index, Middle up (common 'three' gesture)
    elif not thumb_up and index_up and middle_up and ring_up and pinky_up:
        return 4  # All fingers up except thumb
    elif thumb_up and index_up and middle_up and ring_up and pinky_up:
        return 5  # All fingers up

    return "N/A"  # Not recognized or ambiguous gesture


# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple

if not cap.isOpened():
    print("Error: Could not open video stream. Check camera connection or index.")
    exit()

print("Camera opened successfully. Show your hand to recognize digits (0-5). Press 'q' to quit.")

recognized_digit = "N/A"
last_recognition_time = time.time()
display_duration = 1.5  # How long to display a recognized digit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a mirrored view (like a selfie camera)
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB before processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    current_digit = "N/A"

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine if it's a left or right hand
            # MediaPipe provides 'classification' but it's not always available or reliable for 'Left'/'Right' in all versions
            # A more robust way might be to check landmark X positions relative to the wrist (e.g., thumb X vs pinky X)
            # For simplicity, let's assume one hand is being tracked, or use results.multi_handedness if available
            hand_label = "Unknown"
            if results.multi_handedness and hand_idx < len(results.multi_handedness):
                hand_label = results.multi_handedness[hand_idx].classification[0].label

            # Recognize the digit
            current_digit = recognize_digit(hand_landmarks, hand_label)

            # Update the recognized digit for display if changed or if still within display duration
            if current_digit != "N/A" and current_digit != recognized_digit:
                recognized_digit = current_digit
                last_recognition_time = time.time()
            elif current_digit == "N/A" and (time.time() - last_recognition_time > display_duration):
                recognized_digit = "N/A"  # Clear if no hand or no recognized digit for a while

    # Display the recognized digit
    if recognized_digit != "N/A" and (time.time() - last_recognition_time < display_duration):
        text_to_display = f"Digit: {recognized_digit}"
        print(f"Recognized: {recognized_digit}")  # Print to console
    else:
        text_to_display = "Show a digit (0-5)"
        recognized_digit = "N/A"  # Clear the internal state if timed out

    cv2.putText(frame, text_to_display, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Digit Recognizer', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
hands.close()
cap.release()
cv2.destroyAllWindows()