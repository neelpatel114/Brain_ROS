#! /usr/bin/env python3

import socket
import struct
import time
import cv2
import numpy as np
import rospy
import json
import glob
import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

pub = rospy.Publisher('LaneAdjustment', Float64, queue_size=10)
pubSpeed = rospy.Publisher('SpeedAdjustment', Float64, queue_size=10)
command_publisher = rospy.Publisher("/automobile/command", String, queue_size=1)

############################LANE FINDING####################################
class lane_finding:

    def __init__(self):
        # Choose the number of sliding windows
        self.nwindows=8
        self.stop=False
        self.no_lane=False
        self.count=0
        self.margin=20
        self.right_fit_average=0
        
        self.bl=(0,480)
        self.tl=(120,320)
        self.tr=(500,320)
        self.br=(640,480)
        self.sub_image = rospy.Subscriber("/automobile/image_raw", Image, self.image_callback)
        cv2.namedWindow("Image Window", 1)
        print("Hello")
        #self.pub = rospy.Publisher("topic_name", UInt8, queue_size=10)
    
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
        self.show_image(cv_image)
        self.run(cv_image)

    def show_image(self, img): 
        cv2.imshow("Image Window", img)

    def warp(self,img): # mts, dist
        #undist = cv2.undistort(img, mtx, dist, None, mtx)
        img_size = (img.shape[1], img.shape[0])
        #print(img_size)
        offset = 10
        
        # Source points taken from images with straight lane lines, 
        # these are to become parallel after the warp transform
        src = np.float32([
            self.bl, # bottom-left corner
            self.tl, # top-left corner 
            self.tr, # top-right corner
            self.br # bottom-right corner
        ])

        # Destination points are to be parallel, taken into account the image size
        dst = np.float32([
            [offset, img_size[1]],             # bottom-left corner
            [offset, 0],                       # top-left corner
            [img_size[0]-offset, 0],           # top-right corner
            [img_size[0]-offset, img_size[1]]  # bottom-right corner
        ])
        # Calculate the transformation matrix and it's inverse transformation
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        self.binary_warped = cv2.warpPerspective(img, M, img_size)
        
        return M_inv

    def binary_thresholded(self,img):
        # Transform image to gray scale
        gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        # Scale result to 0-255
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        sx_binary = np.zeros_like(scaled_sobel)
        # Keep only derivative values that are in the margin of interest
        sx_binary[(scaled_sobel >= 40) & (scaled_sobel <= 255)] = 1

        # Detect pixels that are white in the grayscale image
        white_binary = np.zeros_like(gray_img)
        white_binary[(gray_img > 200) & (gray_img <= 255)] = 1 #200,255

        # Convert image to HLS
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H = hls[:,:,0]
        S = hls[:,:,2]
        sat_binary = np.zeros_like(S)
        # Detect pixels that have a high saturation value
        sat_binary[(S > 200) & (S <= 255)] = 1 #90 , 255

        hue_binary =  np.zeros_like(H)
        # Detect pixels that are yellow using the hue component
        hue_binary[(H > 10) & (H <= 25)] = 1 #10, 25

        # Combine all pixels detected above
        binary_1 = cv2.bitwise_or(sx_binary, white_binary)
        binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
        binary = cv2.bitwise_or(binary_1, binary_2)
        #plt.imshow(binary, cmap='gray')

        return binary
        
    def canny(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        kernel = 5
        blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
        canny = cv2.Canny(gray, 40, 200)
        return canny

    def region_of_interest(self,canny):
        height = canny.shape[0]
        width = canny.shape[1]
        mask = np.zeros_like(canny)

        triangle = np.array([[
        (150, 380),
        (500, 100),
        (640, 450),]], np.int32)

        cv2.fillPoly(mask, triangle, 255)
        masked_image = cv2.bitwise_and(canny, mask)
        return masked_image

    def average_slope_intercept(self, lines):
        left_fit    = []
        right_fit   = []
        if lines is None:
            return None
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1,x2), (y1,y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < -0.4: # y is reversed in image
                    left_fit.append(slope)
                if slope>0.4:
                    right_fit.append(slope)
        # add more weight to longer lines
        if len(right_fit)==0:
            #print('No left lane detected')
            left_line=[[0,480, 0,380]]
        else:
            self.right_fit_average  = np.average(right_fit)
            #print(self.right_fit_average)
    ### STEP 4: Detection of Lane Lines Using Histogram ###

    def find_lane_pixels_using_histogram(self):
        
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255
        window_img = np.zeros_like(out_img)
     
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set the width of the windows +/- margin
        margin = 30
        # Set minimum number of pixels found to recenter window
        minpix =  30

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #print(nonzerox)
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
            # ## if scan windows added  
            # cv2.rectangle(window_img,(win_xleft_high,win_y_high),(win_xleft_low,win_y_low),(255,255,255),3)
            # cv2.rectangle(window_img,(win_xright_high,win_y_high),(win_xright_low,win_y_low),(255,255,255),3)
            # plt.imshow(window_img)
            # plt.show
            
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        #print(left_lane_inds)
        #print('-------')
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty


    def fit_poly(self,leftx, lefty, rightx, righty):
        ### Fit a second order polynomial to each with np.polyfit() ###
        #print(lefty,leftx,righty,rightx)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)   
        #print(left_fit)

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        
        return left_fit, right_fit, left_fitx,right_fitx,ploty


    ### STEP 5: Detection of Lane Lines Based on Previous Step ###

    def find_lane_pixels_using_prev_poly(self):

        # width of the margin around the previous polynomial to search
        #self.margin = 30
        # Grab activated pixels
        nonzero = self.binary_warped.nonzero()
        #print(np.array(nonzero).shape)
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])    
        ### Set the area of search based on activated x-values ###
        ### within the +/- self.margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (self.prev_left_fit[0]*(nonzeroy**2) + self.prev_left_fit[1]*nonzeroy + 
                        self.prev_left_fit[2] - self.margin)) & (nonzerox < (self.prev_left_fit[0]*(nonzeroy**2) + 
                        self.prev_left_fit[1]*nonzeroy + self.prev_left_fit[2] + self.margin))).nonzero()[0]
        right_lane_inds = ((nonzerox > (self.prev_right_fit[0]*(nonzeroy**2) + self.prev_right_fit[1]*nonzeroy + 
                        self.prev_right_fit[2] - self.margin)) & (nonzerox < (self.prev_right_fit[0]*(nonzeroy**2) + 
                        self.prev_right_fit[1]*nonzeroy + self.prev_right_fit[2] + self.margin))).nonzero()[0]
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty


    ### STEP 6: Calculate Vehicle Position and Curve Radius ###

    def measure_position_meters(self,left_fit, right_fit):
        # Define conversion in x from pixels space to meters
        xm_per_pix = 3.7/1920 * (1280/1920)# meters per pixel in x dimension
        # Choose the y value corresponding to the bottom of the image
        y_max = self.binary_warped.shape[0]
        # Calculate left and right line positions at the bottom of the image
        left_x_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
        right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2] 
        # Calculate the x position of the center of the lane 
        center_lanes_x_pos = (left_x_pos + right_x_pos)//2
        # Calculate the deviation between the center of the lane and the center of the picture
        # The car is assumed to be placed in the center of the picture
        # If the deviation is negative, the car is on the felt hand side of the center of the lane
        veh_pos = (center_lanes_x_pos - (self.binary_warped.shape[1]//2))
        return veh_pos


    ### STEP 7: Project Lane Delimitations Back on Image Plane and Add Text for Lane Info ###

    def project_lane_info(self,img,ploty, left_fitx, right_fitx, M_inv, veh_pos):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Center Line modified
        margin = 60
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        #print(left_fitx[20:40])
        
        pts_left_c = np.array([np.transpose(np.vstack([left_fitx+margin, ploty]))])
        pts_right_c = np.array([np.flipud(np.transpose(np.vstack([right_fitx-margin, ploty])))])
        pts = np.hstack((pts_left_c, pts_right_c))
        
        # Draw the lane onto the warped blank image
        colorwarp_img=cv2.polylines(color_warp, np.int_([pts_left]), False, (0,0, 255),20)
        colorwarp_img=cv2.polylines(color_warp, np.int_([pts_right]), False, (0,0, 255),20)
        colorwarp_img=cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # colorwarp_img=cv2.fillPoly(color_warp, np.int_([pts_i]), (0,0, 255))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
           
        # Combine the result with the original image
        out_img = cv2.addWeighted(img, 0.7, newwarp, 0.3, 0)

        return out_img, newwarp


    ### STEP 8: Lane Finding Pipeline on Video ###

    def lane_finding_pipeline(self,img):
        
        binary_thresh = self.binary_thresholded(img)
        M_inv = self.warp(binary_thresh)

        ## checking ###
        binary_thresh_s = np.dstack((binary_thresh, binary_thresh, binary_thresh))*255
        binary_warped_s = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255
        cv2.imshow('binary',cv2.resize(binary_warped_s,(400,300)))
        
        if self.init:
            print(self.init)
            self.left_fit_hist = np.array([])
            self.right_fit_hist = np.array([])
            self.prev_left_fit = np.array([])
            self.prev_right_fit = np.array([])        

            leftx, lefty, rightx, righty = self.find_lane_pixels_using_histogram()
            if (len(rightx)<150):
                #rightx=[630,631,632,633,634,635,636,636,637,638,631,632,633,634,635,634,631,632,633,634,635,636,637,638,639]
                #righty=[350,355,360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445,450,455,460,465,470]
                rightx=[630,625,626,620,632,635,638,622,623,636,634]
                righty=[430,440,450,461,472,473,474,475,476,469,468]
                print('No right lane detected------------------')
            if (len(leftx)<150):
                leftx=[30,20,10,15,5,11,12,13,14,16,17]
                lefty=[435,450,460,471,472,473,474,475,476,469,468]
                print('No left lane detected--------------------')
            left_fit, right_fit, left_fitx,right_fitx,ploty = self.fit_poly(leftx, lefty, rightx, righty)
            # Store fit in history
            #print(right_fit)
            self.left_fit_hist = np.array(left_fit)
            new_left_fit = np.array(left_fit)
            self.left_fit_hist = np.vstack([self.left_fit_hist, new_left_fit])
            self.right_fit_hist = np.array(right_fit)
            new_right_fit = np.array(right_fit)
            self.right_fit_hist = np.vstack([self.right_fit_hist, new_right_fit])
            self.msg=3
            self.big_slope=[]
        
        elif ((self.stop==False) & (self.init==False)):
            #print(self.stop, self.init)
            self.prev_left_fit = [np.mean(self.left_fit_hist[:,0]), np.mean(self.left_fit_hist[:,1]), np.mean(self.left_fit_hist[:,2])]
            self.prev_right_fit = [np.mean(self.right_fit_hist[:,0]), np.mean(self.right_fit_hist[:,1]), np.mean(self.right_fit_hist[:,2])]
            leftx, lefty, rightx, righty = self.find_lane_pixels_using_prev_poly()
            #if (len(lefty) <150 | len(righty) <150):
                #leftx, lefty, rightx, righty = self.find_lane_pixels_using_histogram()

            if ((len(rightx)<150) & (len(leftx)<150)):
                self.stop=True
            if ((len(rightx)<300) & (len(leftx)!=0)):
                rightx=[631,632,633,634,635,634,631,632,633,634,635,636,637,638,639]
                righty=[400,405,410,415,420,425,430,435,440,445,450,455,460,465,470]
                print('No right lane detected----')
                self.msg=2
            if (len(lefty)!=0):
                if (((max(lefty)-min(lefty))<250) & (0.9 < self.right_fit_average)):
                    print('Stop Line')
                    self.stop=True
            if ((self.stop==False) & (len(leftx)<200) & (len(rightx)!=0)):
                leftx=[23,24,30,20,10,15,5,11,12,13,14,16,17]
                lefty=[430,440,450,460,470,471,472,473,474,475,476,469,468]
                print('No left lane detected----')
                self.msg=1
            print(len(leftx),len(rightx))
            if ((self.stop==False) & (len(rightx)>300) & (len(leftx)>300)):
                #print('99999999')
                self.msg=3
        elif self.stop:
            #self.tr=(600,280)
            #self.br=(640,480)
            print('-----------')
            self.msg=0
            self.big_slope=[]
            
            #self.prev_left_fit = [np.mean(self.left_fit_hist[:,0]), np.mean(self.left_fit_hist[:,1]), np.mean(self.left_fit_hist[:,2])]
            #self.prev_right_fit = [np.mean(self.right_fit_hist[:,0]), np.mean(self.right_fit_hist[:,1]), np.mean(self.right_fit_hist[:,2])]
            leftx, lefty, rightx, righty = self.find_lane_pixels_using_histogram()
            #print(len(leftx), len(rightx))
            #print(max(leftx)-min(leftx))
            
            if ((len(rightx)>6500) & (len(leftx)>6500)):
                self.stop=False
                self.init=True
                return self.frame, None
                #self.tr=(450,280)
                #self.br=(550,480)
            print(self.msg)
            return self.frame, None

        print(self.msg)
        if (len(leftx)==0 or len(rightx)==0):
            return self.frame, None 
        else:
            left_fit, right_fit, left_fitx,right_fitx,ploty = self.fit_poly(leftx, lefty, rightx, righty)
            
        # Add new values to history
        new_left_fit = np.array(left_fit)
        #print(new_left_fit)
        self.left_fit_hist = np.vstack([self.left_fit_hist, new_left_fit])
        new_right_fit = np.array(right_fit)
        self.right_fit_hist = np.vstack([self.right_fit_hist, new_right_fit])
            
        # Remove old values from history
        if (len(self.left_fit_hist) > 4): #10
            self.left_fit_hist = np.delete(self.left_fit_hist, 0,0)
            self.right_fit_hist = np.delete(self.right_fit_hist, 0,0)
                
        veh_pos = self.measure_position_meters(left_fit, right_fit) 
        out_img, newwarp = self.project_lane_info(img, ploty, left_fitx, right_fitx, M_inv, veh_pos)
        return out_img, veh_pos


    def run(self, img):
        cap = cv2.VideoCapture('/Users/npatel/Desktop/Brain_ROS/src/perception/src/LaneCentering/my_video2.h264') # test_sample.mp4
        if not cap.isOpened():
            print('File open failed!')
            cap.release()
            sys.exit()

        ## video out ##
        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) 
        delay=int(1000 / fps)

        angle=0
        result = cv2.VideoWriter('filename3.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (640,480))

        self.init=True
        #mtx, dist = distortion_factors()

        while True:
            ret, frame =cap.read()
            self.frame=cv2.resize(frame,(640,480))
            canny_image = self.canny()
            cropped_canny = self.region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 50, np.array([]), minLineLength=5, maxLineGap=20)
            if lines is not None:
                averaged_lines = self.average_slope_intercept(lines)
            print(frame.shape)

            if not ret:
                break
           
            img_out, angle = self.lane_finding_pipeline(frame)

            #if angle>1.5 or angle <-1.5:
            #   init=True
            #else:
            #    init=False

            self.init=False

            cv2.line(img_out,(self.bl),(self.tl),(255,0,0),2)
            cv2.line(img_out,(self.tl),(self.tr),(255,0,0),2)
            cv2.line(img_out,(self.tr),(self.br),(255,0,0),2)
            cv2.line(img_out,(self.bl),(self.br),(255,0,0),2)
            
            cv2.line(img_out,(320,230),(320,250), (0,0,255),2)
            cv2.putText(img_out,'Publshied msg: '+str(self.msg)[:7],(40,150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6,(255,255,255),2,cv2.LINE_AA)

            
            #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
            cv2.imshow('frame', cv2.resize(img_out,(600,400)))
            #result.write(img_out)
        
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        #result.release()
        cv2.destroyAllWindows()


################################### IMG DATA #####################################
#class ImgCap():
#    def __init__(self):
#        self.sub_image = rospy.Subscriber("/automobile/image_raw", Image, self.image_callback)
#        cv2.namedWindow("Image Window", 1)
#    def image_callback(self, img_msg):
#        # log some info about the image topic
#        #rospy.loginfo(img_msg.header)
#        # Try to convert the ROS Image message to a CV2 Image
#        try:
#            cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
#        except CvBridgeError as e:
#            rospy.logerr("CvBridge Error: {0}".format(e))
#        # Show the converted image
#
#        #comment out if you do not want output
#        #self.show_image(cv_image)
#        self.LaneCentering(cv_image)
#
#    def show_image(self, img): 
#        cv2.imshow("Image Window", img)
#
#    def LaneCentering(self, img):
#        frame = img
#        frame=cv2.resize(frame, (640,480))
#        canny_image = canny(frame)
#        cropped_canny = region_of_interest(canny_image)
#        
#        lines = cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 50, np.array([]), minLineLength=5, maxLineGap=20)
#        #print(lines)
#        if lines is None:
#            print('No line detected')
#            #cv2.imshow("canny", cropped_canny)
#            #fin=frame
#        else:
#            averaged_lines = average_slope_intercept(frame, lines)
#            line_image,mean = display_lines(frame, averaged_lines)
#            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#        #print(combo_image.shape)
#            cv2.line(combo_image,(mean,int(combo_image.shape[0]*4/5)), (mean, int(combo_image.shape[0]*4/5)-10), (0,0,255),2)
#            cv2.line(combo_image,(320,235),(320,245), (0,0,255),2)
#            pos=mean-320
#            #combo_image=cv2.resize(combo_image,(700,500))
#            im0=combo_image
#        #cv2.imshow("final", fin)
#        pub.publish(20.5)
#        #cv2.imshow("line", cropped_canny)
#

################################### MAIN #########################################
#def func():
#        pubSpeed.publish(0.10)
#        time.sleep(1000)
#        pub.publish(18.1)
#        time.sleep(10)
#        pub.publish(-18.1)


if __name__ == '__main__':

    #need the below 4 lines to collect image 
    bridge = CvBridge()
    rospy.init_node('ImgCap_Test', anonymous=True)
    cv2.namedWindow("Image Window", 1)
    PID = "{'action': '4', 'activate': true}"
    PID = PID.replace("'", '"') #must replace '' for json formate (this was easier than regex)
    print(PID)
    command_publisher.publish(PID) #send command to serialNODE
    time.sleep(3)
    #LC = lane_finding()
    lane = lane_finding()

    #loop image collection
    while not rospy.is_shutdown():
        rospy.spin()
        



