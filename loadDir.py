import glob
import sys
import os
import pathlib
from object_detect import *
from lane_detect import *
import torch
import numpy as np
import cv2

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']


def show_yolov7(det, im0):
    for *xyxy, conf, cls in reversed(det):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, im0, label=label, line_thickness=1)


if __name__ == '__main__':

    file_dir_path = sys.argv[1]
    weights = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
        width = 320
        height = 213
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    fileExt = r'.jpg'
    file_name_list = os.listdir(file_dir_path)
    file_name_list.sort(key=lambda x: int(x.split('.')[0]))
    num = 0
    start = time.time()
    for filename in file_name_list:
        with torch.no_grad():
            source_path = os.path.join(file_dir_path, filename)
            source_img = cv2.imread(source_path)
            det = detect(source_path, weights)
            result = laneDetect(source_img)
            #result = source_img.copy()
            show_yolov7(det, result)
            #out.write(result)
            cv2.imshow('source', source_img)
            cv2.imshow('all yolov7 result', result)
            num += 1
            if cv2.waitKey(1) == 27:  # 1 millisecond
                break
    end = time.time()
    seconds = end - start
    fps = num / seconds
    print("Timme")
    print("Time taken : {0} seconds".format(seconds))
    # Calculate frames per second
    print("Estimated frames per second : {0}".format(fps))
    #print(os.path.join(file_dir_path, filename))
