import numpy as np
import cv2
import mediapipe as mp
import pyautogui as pg
import landmarks
import math
import time

#자동종료 방지
pg.FAILSAFE = False

# face mesh를 그리기 위한 객체
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 얼굴  검출을 위한 객체
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 화면 스크린 사이즈 추출
# screen_w, screen_h = pg.size()
screen_w = 2560
screen_h = 1600

# 카메라 실행
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
cap.set(3, screen_w / 5)
cap.set(4, screen_h / 5)
window_name = "MediaPipeFace Mesh"

Sx = None
Sy = None
Sz = None

cnt = 0

Mx, My = 0, 0

avg_ear = 0

with mp_face_mesh.FaceMesh(
  static_image_mode = False, # 부분 얼굴 검출
  max_num_faces=1, #최대 얼굴 개수 1
  refine_landmarks=True, #랜드마크 정교화, 눈동자 주변 랜드마크 추가 출력
  min_detection_confidence=0.5,# 최소 탐지 신뢰도
  min_tracking_confidence=0.5, #최소 추적 신뢰도
) as face_mesh:
  while cap.isOpened():
    
    success, frame = cap.read()

    if not success:
        print("카메라를 찾을 수 없음")
        break
    
    frame.flags.writeable = True # 프레임 속도 줄이기
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 색상 변환
    
    results = face_mesh.process(rgb_frame) # process : 객체 검출

    frame_h, frame_w, _ = frame.shape

    # landmark 캠에 좌표 입력
    if results.multi_face_landmarks: # multi_face_landmarks : 정보 확인
      for face_landmarks in results.multi_face_landmarks:
        landmarks.output_landmarks(results, frame)

        # ear, mar
        left_ear_text, right_ear_text, mouth_text, mar_text, left_ear, right_ear, mar = landmarks.detect_ear_and_mar(results, frame)

        if Sx is None:
          distance, Sx, Sy = landmarks.init_eye_pos(results, frame)
          avg_ear = (left_ear + right_ear) / 2
          pg.moveTo(Sx, Sy)

        flipped_frame = cv2.flip(frame, 1)  # 전체 프레임 반전

        cv2.putText(flipped_frame, mar_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if left_ear_text:
          cv2.putText(flipped_frame, left_ear_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        if right_ear_text:
          cv2.putText(flipped_frame, right_ear_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
          
        # pan값 구하기
        pan = landmarks.get_pan(results, frame)
        cv2.putText(flipped_frame, str(pan), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  
        distance, Cx, Cy = landmarks.init_eye_pos(results, frame)
        cv2.putText(flipped_frame, str(Cx) + ", " + str(Cy), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    mouse_x, mouse_y = pg.position()

    Mx = mouse_x - ((Cx - Sx)  * (distance ** (1/2)))* 0.65
    My = mouse_y + (1.5*(Cy - Sy) * (distance ** (1/2))) * 0.65

    print(f"current eye position: {Cx, Cy}")
    print(f"mouse : {mouse_x, mouse_y}")

    if ((left_ear > avg_ear - 0.04) and (right_ear > avg_ear - 0.04)
        and ~(mouse_x <= 0) and ~(mouse_x >= screen_w) and ~(mouse_y <= 0) and ~(mouse_y >= screen_h)
        and ~(pan > 5)):
      pg.moveTo(Mx, My)
      cnt += 1

    if mar > 0.18:
      pg.click(button='left')
      time.sleep(0.3)

    if cnt == 5:
      Sx = Cx - 0.0000001
      Sy = Cy - 0.0000001
      cnt = 0
    
    cv2.imshow(window_name, flipped_frame)

    if cv2.waitKey(5) & 0xFF == 27: # esc 누르면 종료
      break

cap.release() # 카메라 종료