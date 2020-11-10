# USAGE
# LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3 real_time_ssd_detector.py

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import csv
import pygame
alarm_1 = 'alarm1.mp3'
alarm_2 = 'alarm2.mp3'
from scipy.spatial import distance
import datetime

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] Loading SSD model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] Camera Pi ON ...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

starting_time = time.time()
frame_id = 0
cnt = 0
detection_id = 1

row_list = [["ID","P1" , "P2", "Distance","MSG", "DateTime"]]

#pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(alarm_1)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	frame_id += 1

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	midpoints = []

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		# 0.2 is the minimum probability to filter weak detections
		if confidence > 0.2 :
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			if idx == 15:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				cnt += 1
				midp = (endX, endY)
				midpoints.append([midp, cnt])
				num = len(midpoints)
				print(num)
				# Compute distance between objects
				for m in range(num):
				    for n in range(m+1,num):
				        if m!=n:
				            #GET datetime NOW
				            detection_time = datetime.datetime.now()

				            dst = distance.euclidean(midpoints[m][0], midpoints[n][0])
				            p1 = midpoints[m][1]
				            p2 = midpoints[n][1]
				            print("Distance entre ", p1, " et ", p2, " ==== ", int(dst))
				            if (dst <=200):
				                warning_msg = "ALERT"
				                print("ALERT")
				                
				                pygame.mixer.music.play(-1)
				                time.sleep(1)
				                pygame.mixer.music.stop()
				            else:
				                warning_msg = "normal"
				                print("Normal")
				            
				            # LOG file
				            #save to csv file
				            with open('log.csv', 'w') as file:
				                writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
				                row_list.append([detection_id, p1, p2, int(dst), warning_msg, detection_time])
				                detection_id = detection_id + 1
				                writer.writerows(row_list)

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				# cv2.putText(frame, label, (startX, y),
				#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	elapsed_time = time.time() - starting_time
	fps = frame_id / elapsed_time
	cv2.putText(frame, "FPS=" + str(round(fps,2)), (10,10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
