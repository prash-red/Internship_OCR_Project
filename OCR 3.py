import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List
import sys
import pandas as pd

print(f"Name of the script      : {sys.argv[0]=}")
print(f"Arguments of the script : {sys.argv[1:]=}")


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

dir_str = r"C:\Users\User\Desktop\python\python projects\OCR\DataSample\\"
catalog_file = r'C:\Users\User\Desktop\python\python projects\OCR\DataSample\\Catalog.csv'

df2 = pd.DataFrame(columns=['fname', 'pno'])


for root, dirs, files in os.walk(dir_str):
    for filename in files:
        if (filename.find('jpeg') < 0):
            continue

        fname = dir_str+filename
        print("Find hiden string", fname)
        image = cv2.imread(fname)
        
        base_image = image.copy()
        
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
        #gray = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(gray, (7,7),0)
        
        canny = cv2.Canny(blur, 100, 200)
        thresh = cv2.threshold(canny, 0,225,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(18,18))
        dilate = cv2.dilate(thresh,kernel,iterations = 1)
        
        cnt = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnt[0] if len(cnt) == 2 else cnt[1]
        
        cnt = sorted(cnt, key = lambda x: cv2.boundingRect(x)[1])
        
        #plt.figure(figsize=(10,10))
        #plt.imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
        #plt.show()
         
        #plt.figure(figsize=(10,10))
        #plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)) 
        #plt.show()
        
        #plt.figure(figsize=(10,10))
        #plt.imshow(cv2.cvtColor(dilate, cv2.COLOR_BGR2RGB)) 
        #plt.show()
        
        word = ''
            
        for c in cnt:
            x,y,w,h = cv2.boundingRect(c)
            rect = cv2.rectangle(image , (x,y), (x+w,y+h),(0, 255, 0), 2)
            cropped = base_image[y:y + h, x:x + w]
            
            #plt.figure(figsize=(10,10))
            #plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            #plt.show()
            
            text = pytesseract.image_to_string(cropped)
            if '\x0c' in text:
                num = text.index('\x0c')
                word += text[0:num] + ' '
                
        #print(filename, len(word), word)    
        if(len(word) > 0): 
            df2 = df2.append({'fname':filename, 'pno':word}, ignore_index=True)
            
#print(df2)

#Read the catalog to match the OEM part number 
df = pd.read_csv(catalog_file, header=0, delimiter=',')
k=df[['Oem Number']]
for i in range (0, len(df2)):
    fname = df2.iloc[i, 0]
    pno = df2.iloc[i, 1]
    k = pno.split()
    found = 0
    for c in range (0, len(k)):  
        if (len(k[c]) < 6):
            continue
        l = df[df['Oem Number'] == k[c]]
        if (len(l) > 0) :          
            ono = l.iloc[0, 1]
            applic = l.iloc[0, 2]
            print('found', fname, k[c], ono, applic)
            found = 1
    if (found == 0):
        print('Not found', fname)
        
        

