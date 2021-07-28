# import the necessary packages
import math
import os
import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

IMAGES = 'dataset2/Receipts/'





def deskew1(image):
    # image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
    thresh2 = cv2.bitwise_not(thresh2)
    coords = np.column_stack(np.where(thresh2 > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
        # print("angle < -45: ", angle)
    else:
        # print("angle > -45: ", angle)
        angle = - angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def correctSize(image):

    height = image.shape[1]
    width = image.shape[0]

    print("before height, width ", height, width)

    coef = 0
    if width < 1000 or height < 1000:
        if width < height:
            coef = 1000 / width
        else:
            coef = 1000 / height
        if coef > 2:
            # print("coef: ---", coef)
            coef = 2
        # print("changing size: -------- ", coef)

        image = cv2.resize(image, (int(height * coef), int(width * coef)), interpolation=cv2.INTER_AREA)

    return image








