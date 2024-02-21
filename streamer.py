# -*- encoding: utf-8 -*-
# -------------------------------------------------#
# Date created          : 2020. 8. 18.
# Date last modified    : 2020. 8. 19.
# Author                : chamadams@gmail.com
# Site                  : http://wandlab.com
# License               : GNU General Public License(GPL) 2.0
# Version               : 0.1.0
# Python Version        : 3.6+
# -------------------------------------------------#

import time
import cv2
import imutils
import platform
import numpy as np
from threading import Thread
from queue import Queue
import mediapipe as mp
import random


class Streamer:

    def __init__(self):
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
        print('[wandlab] ', 'OpenCL : ', cv2.ocl.haveOpenCL())

        self.capture = None
        self.thread = None
        self.width = 640
        self.height = 360
        self.stat = False
        self.current_time = time.time()
        self.preview_time = time.time()
        self.sec = 0
        self.Q = Queue(maxsize=128)
        self.started = False
        print("initializing...")
        self.gesture = {
            0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
            6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'mem'
        }
        self.memory_gesture ={0: 'fist', 1: 'one', 11: 'mem'}

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 동그라미 그리기를 위한 변수
        self.first_circle_color = (122, 122, 122)  # 초록색
        self.first_circle_pos = (150, 280)  # 초기 위치
        self.second_circle_color = (122, 122, 122)
        self.second_circle_pos = (500, 280) 

        self.last_update_time = time.time()
        self.update_interval = 5  # 동그라미 위치 업데이트 간격 (초)

        self.first_color = 'gray'
        self.second_color = 'gray'
        
        self.left_hand_action = '?'
        self.right_hand_action = '?'
        
        # Gesture recognition model
        file = np.genfromtxt('data/memory_data.csv', delimiter=',')
        angle = file[:,:-1].astype(np.float32)
        label = file[:, -1].astype(np.float32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    def run(self, src=0):
        self.stop()

        if platform.system() == 'Windows':
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)

        else:
            self.capture = cv2.VideoCapture(src)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.thread is None:
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = False
            self.thread.start()
            print("thread start")

        self.started = True

    def stop(self):

        self.started = False

        if self.capture is not None:
            self.capture.release()
            self.clear()

    def update(self):

        while True:
            try:
                if self.started:
                    (grabbed, frame) = self.capture.read()
                    if grabbed:
                        frame = self.process_frame(frame)
                        self.Q.put(frame)
                    else:
                        print("Failed to grab frame")
            except Exception as e:
                print(f"Exception in update loop: {e}")

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if time.time() - self.last_update_time > self.update_interval:
            # 동그라미 색상 및 위치 무작위 업데이트
            self.first_color = random.choice(['red', 'green', 'blue'])
            self.second_color = random.choice(['red', 'green', 'blue'])
            
            if self.first_color == 'red':
                self.first_circle_color = (255, 0, 0)
            elif self.first_color == 'green':
                self.first_circle_color = (0, 255, 0)
            else:
                self.first_circle_color = (0, 0, 255)
                
            if self.second_color == 'red':
                self.second_circle_color = (255, 0, 0)
            elif self.second_color == 'green':
                self.second_circle_color = (0, 255, 0)
            else:
                self.second_circle_color = (0, 0, 255)
                
            self.last_update_time = time.time()
        
        cv2.circle(frame, self.first_circle_pos, 50, self.first_circle_color, -1)
        cv2.circle(frame, self.second_circle_pos, 50, self.second_circle_color, -1)
        
        results = self.hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_side = 'left' if handedness.classification[0].label == 'Left' else 'right'  # 미디어파이프는 카메라 반전을 고려

                # 각 손의 랜드마크를 기반으로 조인트 각도 계산
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = self.knn.findNearest(data, 3)
                idx = int(results[0][0])

                if idx in self.memory_gesture:
                    action = self.memory_gesture[idx]
                    if hand_side == 'left':
                        self.left_hand_action = action
                        org = (int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0]))
                        cv2.putText(frame, text=self.memory_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    else:
                        self.right_hand_action = action
                        org = (int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0]))
                        cv2.putText(frame, text=self.memory_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # 여기서는 예시로 왼손과 오른손의 동작을 각각 판별한 뒤, 조건을 만족하는지 확인
        # 실제 구현에서는 화면에 표시되는 동그라미의 색상과 위치에 따라 조건을 설정해야 함

        first_correct_action =  (self.first_color == 'red' and self.left_hand_action == 'fist') or \
                                (self.first_color == 'green' and self.left_hand_action == 'one') or \
                                (self.first_color == 'blue' and self.left_hand_action == 'mem')
                                
        second_correct_action = (self.second_color == 'red' and self.right_hand_action == 'fist') or \
                                (self.second_color == 'green' and self.right_hand_action == 'one') or \
                                (self.second_color == 'blue' and self.right_hand_action == 'mem')
                                    

            # Add any additional processing (e.g., gesture recognition) here
        print(first_correct_action, second_correct_action, self.first_color, self.second_color, self.left_hand_action, self.right_hand_action)
    
        if first_correct_action and second_correct_action:
            cv2.putText(frame, "Good!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        return frame
    
    def clear(self):

        with self.Q.mutex:
            self.Q.queue.clear()

    def read(self):

        return self.Q.get()

    def blank(self):

        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)

    def bytescode(self):

        if not self.capture.isOpened():

            frame = self.blank()

        else:

            frame = imutils.resize(self.read(), width=int(self.width))

            if self.stat:
                cv2.rectangle(frame, (0, 0), (120, 30), (0, 0, 0), -1)
                fps = 'FPS : ' + str(self.fps())
                cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        return cv2.imencode('.jpg', frame)[1].tobytes()

    def fps(self):

        self.current_time = time.time()
        self.sec = self.current_time - self.preview_time
        self.preview_time = self.current_time

        if self.sec > 0:
            fps = round(1 / (self.sec), 1)

        else:
            fps = 1

        return fps

    def __exit__(self):
        print('* streamer class exit')
        if self.capture is not None:
            self.capture.release()