import cv2 
from matplotlib import pyplot as plt
import numpy as np
path="E:\Projects\Automatic_Image_enhacement\c3_2501706.jpg"
#"E:\object detection\coding ninjas\deeplearning4\enlight_before-min.jpg"
#image=cv2.imread(path)
def equalize_hist(image):
    image_hsv=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(image_hsv)
    v_e=cv2.equalizeHist(v)
    merged_hsv=cv2.merge((h,s,v_e))
    rgb_image=cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)
    #RGB_image=cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def clahe(image,clipLimit=50):
    original_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(image_hsv)
    
    clahe = cv2.createCLAHE(clipLimit)
    v_clahe=clahe.apply(v)
    clip_hsv=cv2.merge((h,s,v_clahe))
    rgb_image=cv2.cvtColor(clip_hsv, cv2.COLOR_HSV2RGB)
    return rgb_image



def adjust_gamma(image, gamma=1.0):
    	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


#white balance
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result
#face buety deep learning for image enhance
print(help(cv2.ximgproc))