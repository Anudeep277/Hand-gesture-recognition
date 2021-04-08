#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mediapipe as mp
import cv2


# In[3]:


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# In[6]:


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

with mp_holistic.Holistic(min_detection_confidence=0.8) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)     
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=4, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=4, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(173,42,71), thickness=4, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=4, circle_radius=2)
                                 )

                        
        cv2.imshow('webcam feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




