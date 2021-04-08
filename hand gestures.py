#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mediapipe as mp
import cv2


# In[3]:


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# In[26]:


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.8) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)     
        # Recolor image back to BGR for rendering
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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:





# In[16]:


import cv2
import time
import os
import  Handtrackingmodel as ht

wCam,hCam = 1080,720
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

ptime=0

detector=ht.handDetector(detectionCon=0.90)
while True:
    success,img=cap.read()
    image=detector.findHands(img)
    mylist=detector.findPosition(img,draw=False)
    print(mylist)
    
    if len(mylist)!=0:   
        if mylist[8][2]<mylist[8][6]:
           print("index finger open")
           .putText(img,         t(FPS)}",(50,120),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
    ctime=time.time()
    FPS=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f"FPS: {int(FPS)}",(50,120),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
    cv2.imshow("image",img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




