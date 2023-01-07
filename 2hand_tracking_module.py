import cv2
import mediapipe as mp

mpHands = mp.solutions.hands

class Landmark_Track(mpHands.Hands):
    def __init__(self, lm_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = lm_id


    def capture(self, cam = 0):
        cap = cv2.VideoCapture(cam)

        while True:
            ret, frame = cap.read()
            
            if ret:
                self.h, self.w, self.d = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = super().process(rgb_frame)

                # if result.multi_hand_landmarks:
                #     for hand_landmarks in result.multi_hand_landmarks:
                #         for id, lm in enumerate(hand_landmarks.landmark):