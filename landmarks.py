import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui as pg

mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

LEFT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_CENTER = [473]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_CENTER = [468]

LIPS = [62, 81, 13, 311, 291, 402, 14, 178]

PHILTRUM = [164]

ALL_INDEX = LEFT_EYE + LEFT_CENTER + RIGHT_EYE + RIGHT_CENTER + LIPS + PHILTRUM

FACE_CENTER = [6]

screen_w, screen_h = pg.size()

def get_landmarks(results, indices, frame):
    landmarks = []
    frame_h, frame_w, _ = frame.shape

    if not indices:
        return landmarks
    
    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [face_landmarks.landmark[index] for index in indices]
    
    new_landmarks = []

    for landmark in landmarks:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        z = int(landmark.z)
        
        new_landmarks.append((x, y, z))
    
    return new_landmarks

def draw_landmarks(frame, landmarks):
    for landmark in landmarks:
        x, y, _ = landmark
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), cv2.FILLED)

def output_landmarks(results, frame):
    new_landmarks = get_landmarks(results, ALL_INDEX, frame)
    draw_landmarks(frame, new_landmarks)

def calculate_ear(part):
    points = np.array(part, dtype = np.float32)
    # 수직 높이 계산
    vertical_dist = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    vertical_dist /= 2.0
    # 수평 너비 계산
    horizontal_dist = np.linalg.norm(points[0] - points[3])

    # aspect ratio 값 계산
    AR = vertical_dist / horizontal_dist

    return AR

def calculate_mar(part):
    points = np.array(part, dtype = np.float32)
    # 수직 높이 계산
    vertical_dist = np.linalg.norm(points[1] - points[7]) + np.linalg.norm(points[2] - points[6]) + np.linalg.norm(points[3] - points[5])
    vertical_dist /= 2.0
    # 수평 너비 계산
    horizontal_dist = np.linalg.norm(points[0] - points[4])

    # aspect ratio 값 계산
    AR = vertical_dist / horizontal_dist

    return AR

def get_ar_text(ar_value):
    ar_value = float(ar_value)
    if ar_value is not None:
        ar_text = "MAR: {:.2f}".format(ar_value)
    else:
        ar_text = "MAR: N/A"
    return ar_text

def detect_ear_and_mar(results, frame):
    left_ear = calculate_ear(get_landmarks(results, LEFT_EYE, frame))
    right_ear = calculate_ear(get_landmarks(results, RIGHT_EYE, frame))
    #  threshold = avg_ear(left_ear, right_ear)
    ear_text = get_ar_text(left_ear)
    
    mar = calculate_mar(get_landmarks(results, LIPS, frame))
    mar_text = get_ar_text(mar)
    
    left_ear_text = ""
    right_ear_text = ""
    mouth_text = ""

    if left_ear < 0.21:
        left_ear_text = "Left Blink"

    if right_ear < 0.21:
        right_ear_text = "Right Blink"
    
    if mar > 0.35:
        mouth_text = "Yawning"

    return left_ear_text, right_ear_text, mouth_text, mar_text, left_ear, right_ear, mar

def get_pan(results, frame):
    left = get_landmarks(results, LEFT_CENTER, frame)[0]
    left_x, left_y, _ = left
    right = get_landmarks(results, RIGHT_CENTER, frame)[0]
    right_x, right_y, _ = right
    
    philtrum = get_landmarks(results, PHILTRUM, frame)[0]
    philtrum_x, philtrum_y, _ = philtrum

    horizontal_dist = (left_x + right_x) / 2- (philtrum_x)
    vertical_dist = (left_y + right_y) / 2 - (philtrum_y)

    rd = math.sqrt((horizontal_dist ** 2) + (vertical_dist ** 2))

    pan = math.atan(horizontal_dist / rd)

    pan = pan * 180 / math.pi

    return pan


def init_eye_pos(results, frame):
    frame_h, frame_w, _ = frame.shape
    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [face_landmarks.landmark[LEFT_CENTER[0]]]
    left_center_x = int(landmarks[0].x * screen_w)
    left_center_y = int(landmarks[0].y * screen_h)
    
    landmarks = [results.multi_face_landmarks[0].landmark[RIGHT_CENTER[0]]]
    right_center_x = int(landmarks[0].x * screen_w)
    right_center_y = int(landmarks[0].y * screen_h)

    Cx = (right_center_x + left_center_x) / 2
    Cy = (right_center_y + left_center_y) / 2

    distance = ((left_center_x - right_center_x)**2 + (left_center_y - right_center_y)**2)**0.5

    return distance, Cx, Cy
    
