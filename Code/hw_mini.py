'''
Created in September 2018

@author: McKenzie Cheung, Jessica Seto

EC463 Hardware Mini Project -- Vehicle Detection with Raspberry Pi
'''

# Libary imports
import cv2
import numpy as py
import os
import time

from pathlib import Path
# This library can be uncommmented for the user if they want to run the program locally and see the video frames
#from matplotlib import pyplot as plt

# Capture frames from the video clip
clip_cap = cv2.VideoCapture('cars_4.mp4')

# Define video writing
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Save the new file 
hv = cv2.VideoWriter('cars_1_new.avi', fourcc, fps=10, frameSize=(640, 480))

# Trained XML classifiers file for the cars
cars_class = cv2.CascadeClassifier('cars.xml')

# Iteration over frames of the video clip - Loop runs if the frame has been captured
while True: 
    # Read frames
    ret, frames = clip_cap.read()

    # Convert to gray scale to detect cars
    try:
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    except:
        # When video ended, it would throw an error. This helps to overide the error once the video finishes running.
        print("The video has ended.")
        break

    # Detects cars of different sizes
    car_detect = cars_class.detectMultiScale(gray, 1.1, 1)
    print("This is the number of cars: ", len(car_detect))

    # Puts text on the frame
    cv2.putText(img=frames, text="This is the number of cars: ", org=(10,450), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, thickness = 1, color = (255, 255, 255))
    cv2.putText(img=frames, text=str(len(car_detect)), org=(250,450), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, thickness = 1, color = (255, 255, 255))

    # Slows down the frames for better detection - Can be commented out if needed for faster processing
    time.sleep(1.0)

    # Draw rectangle over each car
    num_cars = 0
    for (x, y, w, h) in car_detect:
        cv2.rectangle(frames, (x,y), (x+w, y+h), (0,0,255), 2)
       
    # Display in separate window (Only needed if using matplotlib)
    # cv2.imshow('new_vid', frames)

    hv.write(frames)

    # Break in the code to stop video playing
    if cv2.waitKey(33) == 27:
        break

# De-allocate memory (Only needed if using matplotlib)
# cv2.destroyAllWindows()

hv.release()
