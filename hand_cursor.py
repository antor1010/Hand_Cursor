import cv2
import mediapipe as mp
import pyautogui
import time

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0
thumb_y = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)

            landmarks = hand.landmark
            index_finger = landmarks[8]
            thumb_finger = landmarks[4]

            index_x = int(index_finger.x * frame_width)
            index_y = int(index_finger.y * frame_height)
            thumb_x = int(thumb_finger.x * frame_width)
            thumb_y = int(thumb_finger.y * frame_height)

            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)

            screen_x = int(screen_width / frame_width * index_x)
            screen_y = int(screen_height / frame_height * index_y)

            pyautogui.moveTo(screen_x, screen_y)

            if abs(index_y - thumb_y) < 40:
                pyautogui.click()
                time.sleep(1)

    cv2.imshow('Hand Cursor', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
