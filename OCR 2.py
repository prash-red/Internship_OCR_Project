import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import re
from pytesseract import Output

img = cv2.imread('sample.jpeg')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

custom_config = r'--oem 3 --psm 12'
pytesseract.image_to_string(img, config=custom_config)

img = cv2.imread('sample.jpeg')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TOZERO)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

test = gray = get_grayscale(img)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
plt.show()

test = thresholding(test)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
plt.show()





thresh = thresholding(test)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
plt.show()

opening = opening(test)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(opening, cv2.COLOR_BGR2RGB))
plt.show()

canny = canny(test)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
plt.show()


d = pytesseract.image_to_data(canny, output_type=Output.DICT,config=custom_config)
print(d.keys())

text = pytesseract.image_to_string(canny,config=custom_config)
print(text)

n_boxes = len(d['text'])
for i in range(n_boxes):
    if float(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(canny, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', canny)
cv2.waitKey(0)

