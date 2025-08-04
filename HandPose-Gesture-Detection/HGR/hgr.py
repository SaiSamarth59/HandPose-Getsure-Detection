import cv2
import mediapipe as mp

# Initialize MediaPipe hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Dictionary to hold gesture definitions
gestures = {
    "Open Palm": lambda landmarks: all(landmarks[i].y < landmarks[i-2].y for i in range(8, 21, 4)),
    "Fist": lambda landmarks: all(landmarks[i].y > landmarks[i-2].y for i in range(8, 21, 4)),
    "Thumbs Up": lambda landmarks: landmarks[4].y < landmarks[3].y < landmarks[2].y < landmarks[1].y < landmarks[0].y,
    "Thumbs Down": lambda landmarks: landmarks[4].y > landmarks[3].y > landmarks[2].y > landmarks[1].y > landmarks[0].y,
    "Victory Sign": lambda landmarks: landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y,
    "OK Sign": lambda landmarks: abs(landmarks[4].x - landmarks[8].x) < 0.05 and abs(landmarks[4].y - landmarks[8].y) < 0.05,
    "Pointing Up": lambda landmarks: landmarks[8].y < landmarks[6].y and all(landmarks[i].y > landmarks[i-2].y for i in [12, 16, 20]),
    "Pointing Left": lambda landmarks: landmarks[8].x < landmarks[6].x and all(landmarks[i].x > landmarks[i-2].x for i in [12, 16, 20]),
    "Rock Sign": lambda landmarks: landmarks[4].y < landmarks[3].y and landmarks[8].y < landmarks[6].y and landmarks[20].y < landmarks[18].y,
    "Three Fingers Up": lambda lm: lm[8].y < lm[6].y and lm[12].y < lm[10].y and lm[16].y < lm[14].y and lm[20].y > lm[18].y,

}

# Start the webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Flip image for correct orientation and convert to RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to find hands
        results = hands.process(image_rgb)
        
        # Draw landmarks and classify gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check for each gesture
                for gesture, func in gestures.items():
                    if func(hand_landmarks.landmark):
                        cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        break
        
        # Show the output
        cv2.imshow('Gesture Recognition', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
