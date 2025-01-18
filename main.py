import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


canvas = np.ones((600, 800, 3), dtype="uint8") * 255
objects = [{"rect": (300, 200, 100, 100), "color": (0, 255, 0)}]  
dragging = False
drag_index = -1

def is_inside_object(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    cursor_pos = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get cursor position (index finger tip)
            x, y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, \
                   hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            cursor_pos = int(x * w), int(y * h)

            if cursor_pos:
                cv2.circle(frame, cursor_pos, 10, (255, 0, 0), -1)  # Draw cursor on the screen
                print(f"Cursor Position: {cursor_pos}")

            

            # Detect "pinch" (distance between thumb tip and index finger tip)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinch_distance = np.linalg.norm(
                np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
            )
            pinch_threshold = 0.05  # Adjust based on hand distance in frame

            print(f"Thumb Tip: {thumb_tip.x, thumb_tip.y}, Index Tip: {index_tip.x, index_tip.y}, Distance: {pinch_distance}")

            if pinch_distance < pinch_threshold:
                # Start dragging if cursor is on an object
                if not dragging:
                    for i, obj in enumerate(objects):
                        if is_inside_object(*cursor_pos, obj["rect"]):
                            dragging = True
                            drag_index = i
                            break

                # Update the object's position
                if dragging and drag_index != -1:
                    x, y, _, _ = objects[drag_index]["rect"]
                    w, h = objects[drag_index]["rect"][2:]
                    objects[drag_index]["rect"] = (cursor_pos[0] - w // 2, cursor_pos[1] - h // 2, w, h)
            else:
                dragging = False
                drag_index = -1

    # Draw objects
    for obj in objects:
        cv2.rectangle(frame, obj["rect"][:2], (obj["rect"][0] + obj["rect"][2], obj["rect"][1] + obj["rect"][3]),
                      obj["color"], -1)

    # Show frame
    cv2.imshow("Drag and Drop", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
