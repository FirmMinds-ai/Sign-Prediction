import numpy as np
import shutil
import time
import mediapipe as mp
import cv2
import os
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh, rh])

def preprocess_vedio(path):
    # Set mediapipe model
    path_vedio = path
    lst_frame_MP = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        vidcap = cv2.VideoCapture(path_vedio)
        success,frame = vidcap.read()
        count = -1
        # Loop through each frame
        while success:
            count += 1
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # Export keypoints
            keypoints = extract_keypoints(results)
            lst_frame_MP.append(keypoints)
            success,frame = vidcap.read()

    sequence_length = 30
    number_of_frames = 30

    vidcap = cv2.VideoCapture(path_vedio)
    success,frameTemp = vidcap.read() #FRAME IS IMAGE
    success,frame = vidcap.read()
    count = -1
    actionMagnitude = []
    # Loop through each frame in a single video, to find the action magnitude over each frame in a video
    while success:
        count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        static_back = cv2.cvtColor(frameTemp, cv2.COLOR_BGR2GRAY)
        static_back = cv2.GaussianBlur(static_back, (21, 21), 0)
        diff_frame = cv2.absdiff(static_back, gray)
        actionMagnitude.append(diff_frame.sum())
        frameTemp = np.copy(frame)
        success,frame = vidcap.read() #FRAME IS IMAGE

    #find best n frames to extract************************************
    maxIndex = 0 #best index to start collecting max(n = 30) frame. ex: max=5 means best is to collect frames[5:25]
    maxMagnitude = 0 #for a given number_of_frames
    for i in range(0,len(actionMagnitude)-number_of_frames):
        if sum(actionMagnitude[i:i+number_of_frames]) > maxMagnitude:
            maxIndex = i
            maxMagnitude = sum(actionMagnitude[i:i+number_of_frames])

    lst_frame_MP_FILTERED = []
    #Copy selected frames' keypoints into another location************
    for i in range(maxIndex,maxIndex+number_of_frames):
        lst_frame_MP_FILTERED.append(lst_frame_MP[i])

    window = np.expand_dims(lst_frame_MP_FILTERED,0)
    return window
