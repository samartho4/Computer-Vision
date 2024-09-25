"""
Created by Dr. Mufleh Al-Shatnawi on December 2020.
Copyright (c) 2020 Dr. Mufleh Al-Shatnawi. All rights reserved.

Objective:
    1) Test the Opencv installation by reading video and writing video
"""



from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
###########################################################
import cv2

#####################
inputvideofilepath ="Video1.mpg"
cap = cv2.VideoCapture(inputvideofilepath)
# Check if video file opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object.
# The output is stored in 'outputV1.avi' file.
outputvideofilepath="TestVideo1.avi"
codec = cv2.VideoWriter_fourcc(*'DIVX')
frame_rate = 30 # 10, 20, 25
resolution = (frame_width,frame_height)

out =cv2.VideoWriter(outputvideofilepath, codec , frame_rate, resolution,True)

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    # Our operations on the frame come here
    # Converts to HSV color space, OCV reads colors as BGR
    # frame is converted to hsv
    hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Write the frame into the file 'output.avi'
    out.write(hsvframe)

    # Display the resulting frame
    cv2.imshow('hsvframe',hsvframe)
    # Display the frame
    cv2.imshow('frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
# When everything done, release the video capture and video write objects
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()