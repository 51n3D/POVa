# coding: utf-8
from __future__ import print_function

import numpy as np
import cv2

# This should help with the assignment:
# * Indexing numpy arrays http://scipy-cookbook.readthedocs.io/items/Indexing.html


def parseArguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='Input video file name.')
    parser.add_argument('-i', '--image', help='Input image file name.')
    args = parser.parse_args()
    return args


def image(imageFileName):
    # read image
    img = cv2.imread(imageFileName)
    if img is None:
        print("Error: Unable to read image file", imageFileName)
        exit(-1)
    
    # print image width, height, and channel count
    print("Image dimensions: ", img.shape)

    # Resize to width 400 and height 500 with bicubic interpolation.
    img = cv2.resize(img, (400, 500), interpolation=cv2.INTER_CUBIC)
    
    # Print mean image color and standard deviation of each color channel
    print('Image mean and standard deviation', 
        cv2.meanStdDev(img)[0].flatten(), cv2.meanStdDev(img)[1].flatten())
    
    # Fill horizontal rectangle with color 128.  
    # Position x1=50,y1=120 and size width=200, height=
    rectangle = img.copy()
    rectangle = cv2.rectangle(rectangle, (50, 120), (200, 50), (128, 128, 128), thickness=cv2.FILLED)
    
    # write result to file
    cv2.imwrite('rectangle.png', rectangle)
    
    # Fill every third column in the top half of the image black.
    # The first column sould be black.  
    # The rectangle should not be visible.
    striped = img.copy()
    striped[:striped.shape[0]//2:, ::3] = (0, 0, 0)

    # write result to file
    cv2.imwrite('striped.png', striped)
    
    # Set all pixels with any a value of any collor channel lower than 100 to black (0,0,0).
    clip = img.copy()
    clip[(img < 100).any(axis=2)] = [0, 0, 0]

    #write result to file
    cv2.imwrite('clip.png', clip)
   

def video(videoFileName):
    # open video file and get basic information
    videoCapture = cv2.VideoCapture(videoFileName)    
    frameRate = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not videoCapture.isOpened():
        print("Error: Unable to open video file for reading", videoFileName)
        exit(-1)
    
    # open video file for writing
    videoWriter  = cv2.VideoWriter(
        'videoOut.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 
        frameRate, (frame_width, frame_height))
    if not videoWriter.isOpened():
        print("Error: Unable to open video file for writing", videoFileName)
        exit(-1)
    
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            break;
        
        # Flip image upside down.
        frame = np.flip(frame)
        
        # Add white noise (additive noise with normal distribution).
        # Standard deviation should be 5.
        # use np.random
        frame += np.random.normal(0, 5, frame.shape).astype(np.uint8)
        norm = np.zeros(frame.shape[:-1])
        frame = cv2.normalize(frame, norm, 0, 255, cv2.NORM_MINMAX)
        
        # Add gamma correction.
        # y = x^1.2 -- the image to the power of 1.2
        frame = (frame ** 1.2).astype(np.uint8)

        # Dim blue color to half intensity.
        frame[:, :, 1] = (frame[:, :, 1] / 2).astype(np.uint8)
        
        # Invert colors.
        frame = 255 - frame
        
        # Display the processed frame.
        cv2.imshow("Output", frame)
        # Write the resulting frame to the video file.
        videoWriter.write(frame)
        
        # End the processing on pressing Escape.
        if cv2.waitKey(10) & 0xFF == 27:
            break

        
    cv2.destroyAllWindows()
    videoCapture.release()
    videoWriter.release()


def main():
    args = parseArguments()
    np.random.seed(1)
    image(args.image)
    video(args.video)

if __name__ == "__main__":
    main()

