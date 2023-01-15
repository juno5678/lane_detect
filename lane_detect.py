import numpy as np
import cv2
import math


def ROI(img):
    height = img.shape[0]
    width = img.shape[1]
    triangle = np.array([[(0, int(height * 7 / 8)), (0, height), (width, height), (width, int(height * 7 / 8)), (int(width * 14 / 31), int(height / 8)), (int(width * 16 / 31), int(height / 8))]])
    black_image = np.zeros_like(img)
    # Put the Triangular shape on top of our Black image to create a mask
    mask = cv2.fillPoly(black_image, triangle, (255, 0, 0))
    # applying mask on original image
    masked_image = cv2.bitwise_and(img, mask)
    #cv2.imshow("m", masked_image)
    return masked_image


def find_lines(img, lines, color=[255, 0, 0], thickness=5):
    top = 0
    bottom = 213
    ori_img = img.copy()
    combine_img = img.copy()
    left_x1_set = []
    left_y1_set = []
    left_x2_set = []
    left_y2_set = []
    right_x1_set = []
    right_y1_set = []
    right_x2_set = []
    right_y2_set = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = get_slope(x1, y1, x2, y2)
            angle = math.degrees(math.atan(slope))
            if angle < 0 and angle > -90:
                cv2.line(ori_img, (x1, y1), (x2, y2), [255, 0, 0], thickness)
                # Ignore obviously invalid lines
                if angle > -20:
                    continue
                #print("left slope : %3f , angle : %3f , x1 : %d, y1 : %d, x2 : %d, y2 : %d" % (slope, angle, x1, y1, x2, y2))
                left_x1_set.append(x1)
                left_y1_set.append(y1)
                left_x2_set.append(x2)
                left_y2_set.append(y2)
                cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], thickness)
            elif angle > 0 and angle < 90:
                cv2.line(ori_img, (x1, y1), (x2, y2), [0, 255, 0], thickness)
                # Ignore obviously invalid lines
                if angle < 30. or angle > 80:
                    continue
                #print("right slope : %3f , angle : %3f , x1 : %d, y1 : %d, x2 : %d, y2 : %d" % (slope, angle, x1, y1, x2, y2))
                right_x1_set.append(x1)
                right_y1_set.append(y1)
                right_x2_set.append(x2)
                right_y2_set.append(y2)
                cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], thickness)
    try:
        avg_right_x1 = int(np.mean(right_x1_set))
        avg_right_y1 = int(np.mean(right_y1_set))
        avg_right_x2 = int(np.mean(right_x2_set))
        avg_right_y2 = int(np.mean(right_y2_set))
        right_slope = get_slope(avg_right_x1, avg_right_y1, avg_right_x2, avg_right_y2)

        right_y1 = top
        right_x1 = int(avg_right_x1 + (right_y1 - avg_right_y1) / right_slope)
        right_y2 = bottom
        right_x2 = int(avg_right_x1 + (right_y2 - avg_right_y1) / right_slope)
        cv2.line(combine_img, (right_x1, right_y1), (right_x2, right_y2), [0, 255, 0], thickness)
    except ValueError:
        # Don't error when a line cannot be drawn
        pass

    try:
        avg_left_x1 = int(np.mean(left_x1_set))
        avg_left_y1 = int(np.mean(left_y1_set))
        avg_left_x2 = int(np.mean(left_x2_set))
        avg_left_y2 = int(np.mean(left_y2_set))
        left_slope = get_slope(avg_left_x1, avg_left_y1, avg_left_x2, avg_left_y2)

        left_y1 = top
        left_x1 = int(avg_left_x1 + (left_y1 - avg_left_y1) / left_slope)
        left_y2 = bottom
        left_x2 = int(avg_left_x1 + (left_y2 - avg_left_y1) / left_slope)
        cv2.line(combine_img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)
    except ValueError:
        # Don't error when a line cannot be drawn
        pass

    right_line = [right_x1, right_y1, right_x2, right_y2]
    left_line = [left_x1, left_y1, left_x2, left_y2]
    #cv2.imshow('ori_line', ori_img)
    #cv2.imshow('combine_line', combine_img)
    return right_line, left_line


def get_slope(x1, y1, x2, y2):
    return ((y2 - y1) / (x2 - x1))


def draw_result(input_img, right_line, left_line):

    result_img = input_img.copy()
    Lm = get_slope(left_line[0], left_line[1], left_line[2], left_line[3])
    Lc = left_line[1] - Lm * left_line[0]
    Rm = get_slope(right_line[0], right_line[1], right_line[2], right_line[3])
    Rc = right_line[1] - Rm * right_line[0]
    #print("Lm : %3f , Lc : %3f , Rm : %3f , Rc : %3f " % (Lm, Lc, Rm, Rc))
    Lbottomy = input_img.shape[0]
    Lupy = int(Lbottomy * 2.3 / 5)
    Lbottomx = int((Lbottomy - Lc) / Lm)
    Lupx = int((Lupy - Lc) / Lm)

    Rbottomy = input_img.shape[0]
    Rupy = int(Rbottomy * 2.3 / 5)
    Rbottomx = int((Rbottomy - Rc) / Rm)
    Rupx = int((Rupy - Rc) / Rm)

    road_points = np.array([[Lupx, Lupy], [Lbottomx, Lbottomy], [Rbottomx, Rbottomy], [Rupx, Rupy]])
    road_img = result_img.copy()
    road_img = cv2.fillConvexPoly(road_img, road_points, (0, 200, 0))
    result_img = cv2.addWeighted(result_img, 1, road_img, 0.3, 10)
    cv2.line(result_img, (Lbottomx, Lbottomy), (Lupx, Lupy), [255, 0, 0], 3)
    cv2.line(result_img, (Rbottomx, Rbottomy), (Rupx, Rupy), [255, 0, 0], 3)

    return result_img


def hough_lines(img):

    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_len = 10
    max_line_gap = 45
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    right_line, left_line = find_lines(line_img, lines)
    return right_line, left_line


def laneDetect(input_img):
    result_img = input_img.copy()
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 120)
    masked = ROI(edges)
    right_line, left_line = hough_lines(masked)
    result_img = draw_result(input_img, right_line, left_line)

    #cv2.imshow('line', line_image)
    #cv2.imshow('gray', gray)
    #cv2.imshow('blurred', blurred)
    #cv2.imshow('canny', edges)
    #cv2.imshow('masked', masked)
    #cv2.imshow('result_img', result_img)
    return result_img
