# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:57:06 2018

@author: tanzi
"""

# importing libraries
import cv2

# loading cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Specify your custom image path here
image_path = "bb_group.jpeg"

# Function to detect face image
# It will detect face, draw rectangle around that face.
# Then it will find eye in the face and draw rectabgle around it
def detect_face(image_path):
    # read the image
    image = cv2.imread(image_path)
    # make the image grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # getting co ordinate of upper left corner, and weidth & height of the rectengle
    # i.e. detecting the face
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    
    # for loop to itterate through faces and draw an rectangle
    for (x_f, y_f, width, height) in faces:
        cv2.rectangle(image, (x_f,y_f), (x_f + width, y_f+ height), (255, 0, 0), 2)
        
        # now we will detect eye inside the detected face to save the world(kidding! saving computational power, time)
        # we will specify area for both color and grayscale image as face will be detected in grayscale and rectangle will be drawn on color image
        gray_sub_image = gray_image[y_f:y_f + height, x_f:x_f + width]
        rgb_sub_image = image[y_f:y_f + height, x_f:x_f + width]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray_sub_image, 1.1, 5)
        
        # drawing rectangle around eye
        for (x_e, y_e, width_e, height_e) in eyes:
            cv2.rectangle(rgb_sub_image, (x_e, y_e), (x_e + width_e, y_e + height_e), (0, 255, 0), 2)
    return image

image = detect_face(image_path)
cv2.imshow('Faces Found', image)
cv2.waitKey(0)
