#! /usr/bin/env python3

import socket
import struct
import time
import cv2
import numpy as np
import rospy
import json

from cv_bridge       import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

pub = rospy.Publisher('LaneAdjustment', Float64, queue_size=10)

def make_points(image, line):
    if line is None:
        return None
    slope, intercept = line
    #print(slope)
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    #if x1 or x2>640:
     #   return None
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -0.2: # y is reversed in image
                left_fit.append((slope, intercept))
            if slope>0.4:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    if len(left_fit)==0:
        #print('No left lane detected')
        left_line=[[0,480, 0,380]]
    else:
        left_fit_average  = np.average(left_fit, axis=0)
        left_line  = make_points(image, left_fit_average)
        
    if len(right_fit)==0:
        #print('No right lane detected')
        right_line=[[640,480, 640,380]]
    else:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    
    averaged_lines = [left_line, right_line]
    return averaged_lines
    
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        p=lines[0][0][2]
        q=lines[1][0][2]
        #print(p,q)
        mean=int((p+q)/2)
        for line in lines:
            for x1, y1, x2, y2 in line:
                #print(line)
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image, mean
    
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 40, 200)
    return canny

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([[
    (0, 430),
    (300, 250),
    (width, height),]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image





class ImgCap():
    def __init__(self):
        self.sub_image = rospy.Subscriber("/automobile/image_raw", Image, self.image_callback)
        cv2.namedWindow("Image Window", 1)
    def image_callback(self, img_msg):
        # log some info about the image topic
        #rospy.loginfo(img_msg.header)
        # Try to convert the ROS Image message to a CV2 Image
        try:
            cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        # Show the converted image

        #comment out if you do not want output
        #self.show_image(cv_image)
        self.LaneCentering(cv_image)

    def show_image(self, img): 
        cv2.imshow("Image Window", img)
        cv2.waitKey(3)

    def LaneCentering(self, img):
        frame = img
        frame=cv2.resize(frame, (640,480))
        canny_image = canny(frame)
        cropped_canny = region_of_interest(canny_image)
        
        lines = cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 50, np.array([]), minLineLength=5, maxLineGap=20)
        #print(lines)
        if lines is None:
            print('No line detected')
            #cv2.imshow("canny", cropped_canny)
            #fin=frame
        else:
            averaged_lines = average_slope_intercept(frame, lines)
            line_image,mean = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        #print(combo_image.shape)
            cv2.line(combo_image,(mean,int(combo_image.shape[0]*4/5)), (mean, int(combo_image.shape[0]*4/5)-10), (0,0,255),2)
            cv2.line(combo_image,(320,235),(320,245), (0,0,255),2)
            pos=mean-320
            #combo_image=cv2.resize(combo_image,(700,500))
            im0=combo_image
        #cv2.imshow("final", fin)
        pub.publish(20.5)
        #cv2.imshow("line", cropped_canny)

if __name__ == '__main__':

    #need the below 4 lines to collect image 
    bridge = CvBridge()
    rospy.init_node('ImgCap_Test', anonymous=True)
    cv2.namedWindow("Image Window", 1)
    ImgCap_object = ImgCap()

    #loop image collection
    while not rospy.is_shutdown():
        rospy.spin()


