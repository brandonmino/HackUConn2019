import time
import os
import cv2
import numpy as np
import sys
import speech_recognition as sr

def main():

    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    while True:

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        maxArea = 0

        for (_x,_y,_w,_h) in faces:
            if  _w*_h > maxArea:
                x = _x
                y = _y
                w = _w
                h = _h
                maxArea = w*h

    #If one or more faces are found, draw a rectangle around the
    #largest face present in the picture
        if maxArea > 0 :
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Define font to use for text
        font = cv2.FONT_HERSHEY_DUPLEX

        # Write last_emotion to bottom left of frame
        cv2.putText(frame, "TEXT", (80, 460), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    # Count and list for determining mode of recent emotions

# Execute main() if executed
if __name__ == "__main__":
    main()
