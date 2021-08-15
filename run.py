import cv2
import skimage.io
import mediapipe as mp
import argparse
from glob import glob
import os
import math
import numpy as np
from numpy.core.defchararray import find
from numpy.lib.function_base import disp

def extractCoordinates(results, landmark_number):
    x = int(results.multi_face_landmarks[0].landmark[landmark_number].x * 1024)
    y = int(results.multi_face_landmarks[0].landmark[landmark_number].y * 1024)
    return [x, y]

def drawCircle(image, location):
    cv2.circle(image, location, 5, (0, 0, 255), -1)

def drawPolylines(image, pts):
    cv2.polylines(image, [pts], isClosed = False, color = (0, 255, 0), thickness = 3)

def drawArrow(image, start, end):
    cv2.arrowedLine(image, start, end, (0, 255, 0), 2)
    cv2.arrowedLine(image, end, start, (0, 255, 0), 2)

def findDistance(start, end):
    x1, y1 = start
    x2, y2 = end
    return (((x2 - x1)**2) + ((y2 - y1)**2))**(1/2)

def findDistance_poly(pts):
    jaw_distance = 0
    for i in range(len(pts) - 1):
        jaw_distance += findDistance(pts[i], pts[i+1])
    return jaw_distance

def angle_trunc(a):
	while a < 0.0:
		a += math.pi * 2
	return a

def findAngle(start, end):
    x1, y1 = start
    x2, y2 = end
    deltaY = y2 - y1
    deltaX = x2 - x1
    return math.degrees(angle_trunc(math.atan2(deltaY, deltaX)))

def main(image):

    # setting config for facemesh
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=3,
        min_detection_confidence=0.3) as face_mesh:
        image = skimage.io.imread(image)
        image = cv2.resize(image, (1024, 1024))
        
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(image)
        
    keypoints_mapping = {"left_ear": 234, 
    "right_ear": 454, 
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], 
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], 
    "jaw_line": [234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323, 454],
    "upper_head": [10, 8],
    "middle_head": [8, 1],
    "bottom_head": [164, 152],
    "left_ear_to_nose": [234, 5],
    "nose_to_right_ear": [5, 454],
    "left_eyebrow": 107,
    "right_eyebrow": 336, 
    "left_corner_of_mouth": 61, 
    "right_corner_of_mouth": 291}
    
    clone = image.copy()
    keypoints = image.copy()
    distance_dict = {}
    angle_dict = {}

    # keypoints
    for i in range(368):
        location = extractCoordinates(results, i)
        drawCircle(keypoints, location)

    # ear to ear
    left_ear = extractCoordinates(results, keypoints_mapping["left_ear"])
    right_ear = extractCoordinates(results, keypoints_mapping["right_ear"])
    drawCircle(clone, left_ear)
    drawCircle(clone, right_ear)
    drawArrow(clone, left_ear, right_ear)
    ear_ear_distance = findDistance(left_ear, right_ear)
    ear_ear_angle = findAngle(left_ear, right_ear)
    distance_dict["ear_to_ear_distance"] = ear_ear_distance
    angle_dict["ear_to_ear_angle"] = ear_ear_angle

    # eye to eye
    left_eye_1 = extractCoordinates(results, keypoints_mapping["left_eye"][3])
    left_eye_2 = extractCoordinates(results, keypoints_mapping["left_eye"][5])
    left_eye_3 = extractCoordinates(results, keypoints_mapping["left_eye"][11])
    left_eye_4 = extractCoordinates(results, keypoints_mapping["left_eye"][13])
    right_eye_1 = extractCoordinates(results, keypoints_mapping["right_eye"][3])
    right_eye_2 = extractCoordinates(results, keypoints_mapping["right_eye"][5])
    right_eye_3 = extractCoordinates(results, keypoints_mapping["right_eye"][11])
    right_eye_4 = extractCoordinates(results, keypoints_mapping["right_eye"][13])
    left_eye = [(left_eye_1[0] + left_eye_2[0] + left_eye_3[0] + left_eye_4[0])//4, (left_eye_1[1] + left_eye_2[1] + left_eye_3[1] + left_eye_4[1])//4]
    right_eye = [(right_eye_1[0] + right_eye_2[0] + right_eye_3[0] + right_eye_4[0])//4, (right_eye_1[1] + right_eye_2[1] + right_eye_3[1] + right_eye_4[1])//4]
    drawCircle(clone, left_eye)
    drawCircle(clone, right_eye)
    drawArrow(clone, left_eye, right_eye)
    eye_eye_distance = findDistance(left_eye, right_eye)
    eye_eye_angle = findAngle(left_eye, right_eye)
    distance_dict["eye_to_eye_distance"] = eye_eye_distance
    angle_dict["eye_to_eye_angle"] = eye_eye_angle
    
    # eyebrow center to chin
    left_eyebrow = extractCoordinates(results, keypoints_mapping["left_eyebrow"])
    right_eyebrow = extractCoordinates(results, keypoints_mapping["right_eyebrow"])
    chin = extractCoordinates(results, keypoints_mapping["bottom_head"][1])
    center_eyebrow = [(left_eyebrow[0] + right_eyebrow[0])//2, (left_eyebrow[1] + right_eyebrow[1])//2]
    drawCircle(clone, center_eyebrow)
    drawCircle(clone, chin)
    drawArrow(clone, center_eyebrow, chin)
    eyebrow_chin_distance = findDistance(center_eyebrow, chin)
    eyebrow_chin_angle = findAngle(center_eyebrow, chin)
    distance_dict["eyebrow_to_chin_distance"] = eyebrow_chin_distance
    angle_dict["eyebrow_to_chin_angle"] = eyebrow_chin_angle

    # jawline
    pts = []
    for item in keypoints_mapping["jaw_line"]:
        current = extractCoordinates(results, item)
        pts.append(current)
    pts = np.array(pts)
    drawPolylines(clone, pts)
    jawline_distance = findDistance_poly(pts)
    distance_dict["jawline_distance"] = jawline_distance

    # left mouth cornet to right mouth corner
    left_mouth = extractCoordinates(results, keypoints_mapping["left_corner_of_mouth"])
    right_mouth = extractCoordinates(results, keypoints_mapping["right_corner_of_mouth"])
    drawCircle(clone, left_mouth)
    drawCircle(clone, right_mouth)
    drawArrow(clone, left_mouth, right_mouth)
    left_right_mouth_distance = findDistance(left_mouth, right_mouth)
    left_right_mouth_angle = findAngle(left_mouth, right_mouth)
    distance_dict["left_right_mouth_distance"] = left_right_mouth_distance
    angle_dict["left_right_mouth_angle"] = left_right_mouth_angle

    # virtual line
    y_added = int((5*clone.shape[1])/100)
    x_added = int((10*clone.shape[1])/100)
    left_virtualline = [left_ear[0]-x_added, chin[1]+y_added]
    right_virtualline = [right_ear[0]+x_added, chin[1]+y_added]
    drawCircle(clone, left_virtualline)
    drawCircle(clone, right_virtualline)
    drawArrow(clone, left_virtualline, right_virtualline)
    virtualline_distance = findDistance(left_virtualline, right_virtualline)
    virtualline_angle = findAngle(left_virtualline, right_virtualline)
    distance_dict["virtualline_distance"] = virtualline_distance
    angle_dict["virtualline_angle"] = virtualline_angle

    # left eyeball
    left_eye_1 = extractCoordinates(results, keypoints_mapping["left_eye"][12])
    left_eye_2 = extractCoordinates(results, keypoints_mapping["left_eye"][4])
    left_eyeball_horizontal_distance = findDistance(left_eye_1, left_eye_2)
    left_eyeball_horizontal_angle = findAngle(left_eye_1, left_eye_2)
    distance_dict["left_eyeball_horizontal_distance"] = left_eyeball_horizontal_distance
    angle_dict["left_eyeball_horizontal_angle"] = left_eyeball_horizontal_angle
    left_eye_3 = extractCoordinates(results, keypoints_mapping["left_eye"][3])
    left_eye_4 = extractCoordinates(results, keypoints_mapping["left_eye"][5])
    left_eye_5 = extractCoordinates(results, keypoints_mapping["left_eye"][11])
    left_eye_6 = extractCoordinates(results, keypoints_mapping["left_eye"][13])   
    center_left_eye_1 = [(left_eye_3[0]+left_eye_6[0])//2, (left_eye_3[1]+left_eye_6[1])//2]
    center_left_eye_2 = [(left_eye_4[0]+left_eye_5[0])//2, (left_eye_4[1]+left_eye_5[1])//2]
    left_eyeball_vertical_distance = findDistance(center_left_eye_1, center_left_eye_2)
    left_eyeball_vertical_angle = findAngle(center_left_eye_1, center_left_eye_2)
    distance_dict["left_eyeball_vertical_distance"] = left_eyeball_vertical_distance
    angle_dict["left_eyeball_vertical_angle"] = left_eyeball_vertical_angle

    right_eye_1 = extractCoordinates(results, keypoints_mapping["right_eye"][12])
    right_eye_2 = extractCoordinates(results, keypoints_mapping["right_eye"][4])
    right_eyeball_horizontal_distance = findDistance(right_eye_1, right_eye_2)
    right_eyeball_horizontal_angle = findAngle(right_eye_1, right_eye_2)
    distance_dict["right_eyeball_horizontal_distance"] = right_eyeball_horizontal_distance
    angle_dict["right_eyeball_horizontal_angle"] = right_eyeball_horizontal_angle
    right_eye_3 = extractCoordinates(results, keypoints_mapping["right_eye"][3])
    right_eye_4 = extractCoordinates(results, keypoints_mapping["right_eye"][5])
    right_eye_5 = extractCoordinates(results, keypoints_mapping["right_eye"][11])
    right_eye_6 = extractCoordinates(results, keypoints_mapping["right_eye"][13])   
    center_right_eye_1 = [(right_eye_3[0]+right_eye_6[0])//2, (right_eye_3[1]+right_eye_6[1])//2]
    center_right_eye_2 = [(right_eye_4[0]+right_eye_5[0])//2, (right_eye_4[1]+right_eye_5[1])//2]
    right_eyeball_vertical_distance = findDistance(center_right_eye_1, center_right_eye_2)
    right_eyeball_vertical_angle = findAngle(center_right_eye_1, center_right_eye_2)
    distance_dict["right_eyeball_vertical_distance"] = right_eyeball_vertical_distance
    angle_dict["right_eyeball_vertical_angle"] = right_eyeball_vertical_angle

    # uppperhead
    upperhead_1 = extractCoordinates(results, keypoints_mapping["upper_head"][0])
    upperhead_2 = extractCoordinates(results, keypoints_mapping["upper_head"][1])
    upperhead_distance = findDistance(upperhead_1, upperhead_2)
    upperhead_angle = findAngle(upperhead_1, upperhead_2)
    distance_dict["upperhead_distance"] = upperhead_distance
    angle_dict["upperhead_angle"] = upperhead_angle

    # middlehead
    middlehead_1 = extractCoordinates(results, keypoints_mapping["middle_head"][0])
    middlehead_2 = extractCoordinates(results, keypoints_mapping["middle_head"][1])
    middlehead_distance = findDistance(middlehead_1, middlehead_2)
    middlehead_angle = findAngle(middlehead_1, middlehead_2)
    distance_dict["middlehead_distance"] = middlehead_distance
    angle_dict["middlehead_angle"] = middlehead_angle

    # bottomhead
    bottomhead_1 = extractCoordinates(results, keypoints_mapping["bottom_head"][0])
    bottomhead_2 = extractCoordinates(results, keypoints_mapping["bottom_head"][1])
    bottomhead_distance = findDistance(bottomhead_1, bottomhead_2)
    bottomhead_angle = findAngle(bottomhead_1, bottomhead_2)
    distance_dict["bottomhead_distance"] = bottomhead_distance
    angle_dict["bottomhead_angle"] = bottomhead_angle

    # leftear_nose
    left_ear = extractCoordinates(results, keypoints_mapping["left_ear_to_nose"][0])
    nose = extractCoordinates(results, keypoints_mapping["left_ear_to_nose"][1])
    left_ear_nose_distance = findDistance(left_ear, nose)
    left_ear_nose_angle = findAngle(left_ear, nose)
    distance_dict["left_ear_nose_distance"] = left_ear_nose_distance
    angle_dict["left_ear_nose_angle"] = left_ear_nose_angle

    # rightear_nose
    right_ear = extractCoordinates(results, keypoints_mapping["nose_to_right_ear"][0])
    nose = extractCoordinates(results, keypoints_mapping["nose_to_right_ear"][1])
    right_ear_nose_distance = findDistance(right_ear, nose)
    right_ear_nose_angle = findAngle(right_ear, nose)
    distance_dict["right_ear_nose_distance"] = right_ear_nose_distance
    angle_dict["right_ear_nose_angle"] = right_ear_nose_angle

    return distance_dict, angle_dict, clone, keypoints