import cv2
import matplotlib.pyplot as plt
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not
# importing the necessary libraries

mask = './mask_1920_1080.png'
video_path = './data/parking_1920_1080.mp4' 
# mask.png and video.mp4 is already in our data set and our project directory. 

mask = cv2.imread(mask, 0)
# we read our mask png and turned it to a numpy array

cap = cv2.VideoCapture(video_path)
# we captured our videp.mp4 and took it to a frame 

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
# we getting the connected componens with the mask.png 
# 4 or 8: used for components with 4 or 8 connections. 
# 4 only considers up, down, left and right neighbors
# cv_325 is a return type of our data

spots = get_parking_spots_bboxes(connected_components)
# using the method in the util.py class
# for gettin the coordinates 

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))
# this function calculates the average brightness difference between two images 
# simple and effective method to understand how different two images are.

step = 30 # operation will be performed every 30th frame of the video
spots_status = [None for j in spots] # a list that holding the parking spots situations (empty or not)
diffs = [None for j in spots] # list that keeps the brightness differences 
previous_frame = None # holding the previous frame for comparison 

frame_nmr = 0 # frame number
ret = True
while ret: # loop continues as the video is processed frame by frame
    ret, frame = cap.read()
    
    if not ret or frame is None:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        # In this for loop, we can actually see the relationship between artificial neural networks and mathematics. 
        # we calculate the coordinate difference for each frame one by one. 
        # The difference is calculated at each step (30th frame).
        # spot_crop: Crop of each parking area (with coordinates obtained from the mask) from the current frame in the video.
        # calc_diff(): Calculates the brightness difference between the previous and current frames of the parking area and saves it in the diffs list.
        
        print([diffs[j] for j in np.argsort(diffs)][::-1])
        # plt.figure()
        # plt.hist([diffs[j] / np.amax() for j in np.argsort(diffs)][::-1])
        # if frame_nmr == 300:
        #     plt.show()

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in range(len(diffs)) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status
        # arr_: control list that update the parking slots according to difference ratio 
        # empty_or_not(): checking every parking slot that if it is empty or not and adding to spots_status list
    
    if frame_nmr % step ==0:
        previous_frame = frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
        # the status of parking spaces (empty/occupied) is marked with green or red rectangles
        
    cv2.rectangle(frame, (80,20), (550, 80), (0,0,0), -1)

    cv2.putText(frame, 'Avaible spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr = frame_nmr + 1
    # shows the parking slots in the screen. we can see the empty parking slots' numbers 
    # we can see the video with openCV, and if the user press the q button, it will stop 

cap.release()
cv2.destroyAllWindows()
# and we released all the parking slots and closed all windows
