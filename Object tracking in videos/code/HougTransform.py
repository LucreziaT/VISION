import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import defaultdict
import math
from scipy import ndimage
from skimage import filters

roi_defined = False
threshold = 20

def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked,
	# record the starting ROI coordinates
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)
		roi_defined = True

def calcGrad_Ori(frame, th):
	#Calculating the gradient module:
	img = frame[:, :, 2]
	g_x,g_y = np.gradient(img)
	grad = np.sqrt(g_x**2 + g_y**2)
	
	#Calculating the real orientation by arctan and finding not valids index
	orientation = np.arctan2(g_y, g_x)
	notvalidIndex = np.where(grad < threshold)
	validIndex = np.where(grad > threshold)

	#Turning red all points not used
	oriValid = cv2.cvtColor(np.float32(orientation), cv2.COLOR_GRAY2BGR)
	oriValid = cv2.normalize(oriValid, oriValid, 0, 255, cv2.NORM_MINMAX)
	oriValid = np.uint8(oriValid)
	oriValid[notvalidIndex[0], notvalidIndex[1], :] = [0, 0, 255]

	return (grad-grad.min())/(float)(grad.max() - grad.min()), orientation, oriValid, notvalidIndex, validIndex

def calcHough(tHough, orientation, rTable, validIndex):
	indice = orientation*90//math.pi

	for px, py in zip(validIndex[0], validIndex[1]):
		idx = indice[px, py]
		if idx in rTable:
			for value in rTable[idx]:
				if (py + value[0] >= 0 and py + value[0] < tHough.shape[1]) and (px + value[1] >= 0 and px + value[1] < tHough.shape[0]):
					tHough[px + value[1], py + value[0]] += 1.

	return tHough

# cap = cv2.VideoCapture('Test-Videos/Antoine_Mug.mp4')
cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')
# cap = cv2.VideoCapture('Test-Videos/VOT-Basket.mp4')
# cap = cv2.VideoCapture(Test-Videos/VOT-Car.mp4')
# cap = cv2.VideoCapture('Test-Videos/VOT-Sunshade.mp4')
# cap = cv2.VideoCapture('Test-Videos/VOT-Woman.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else :
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break

track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


#Creating R-Table
rTable = defaultdict(list)

#Populating the table
gradient, orientation, ori, _, validIndex= calcGrad_Ori(hsv_roi, threshold)
roiCenter = np.array([int(r + (h//2)), int(c + (w//2))])
orientation = orientation*90//math.pi

#Populating rTable
for px,py in zip(validIndex[0], validIndex[1]):
	distance = roiCenter - np.array([py+r, px+c])
	rTable[orientation[px, py]].append(distance)

cpt = 1
while(1):
	ret ,frame = cap.read()
	if ret == True:
		fram_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		gradient, orientation , ori, _, validIndex= calcGrad_Ori(fram_hsv, threshold)
		tHough = np.zeros((orientation.shape))

		tHough = calcHough(tHough, orientation, rTable, validIndex)

		# Argmax:
		cy, cx = np.unravel_index(np.argmax(tHough), tHough.shape)
		r, c = max(cx - (h // 2), 0), max(cy - (w // 2), 0)

		# Draw a blue rectangle on the current image and normalize Hough
		frame_tracked = cv2.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 2)
		tHough = np.uint8(tHough)
		tHough = cv2.normalize(tHough, tHough, 0, 255, cv2.NORM_MINMAX)

		#Plotting all images
		cv2.imshow('Sequence', frame_tracked)
		cv2.imshow('Valid orientation', ori)
		cv2.imshow("Orientation / argument", orientation)
		cv2.imshow("Gradient magnitude", gradient)
		cv2.imshow("Transformee Hough", tHough)

		k = cv2.waitKey(60) & 0xff
		if k == 27:
				break
		elif k == ord('s'):
				cv2.imwrite('./images/q4/Q4_Frame_%04d.png'%cpt,frame_tracked)
				cv2.imwrite('./images/q4/Q4_Frame_Hough_%04d.png'%cpt,tHough)
				cv2.imwrite('./images/q4/Q4_Frame_Ori_%04d.png'%cpt,ori)
		cpt += 1
	else:
		break

cv2.destroyAllWindows()
cap.release()
