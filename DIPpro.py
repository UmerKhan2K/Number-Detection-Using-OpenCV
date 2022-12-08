#!/usr/bin/env python
# coding: utf-8

# 

# In[2]:


# Required imports  
from IPython import get_ipython
from matplotlib.pyplot import imread, imshow, imsave
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from PIL import Image
# RGB to Gray image

file = input('Enter file name')
print(file)
file = file+'.jpeg'
print(file)

img = Image.open(file)
imgGray = img.convert('L')
imgGray.save('test_gray.jpeg')


# Canny Edge Detection

import cv2
from PIL import Image as im
# Read the original image
img = cv2.imread(file) 
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

 
    
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image

cv2.imshow('Canny Edge Detection', edges)
print(edges.shape)
data = im.fromarray(edges)
data.save('C_edges.jpeg')



cv2.waitKey(0)
 
cv2.destroyAllWindows()










# Croping Image
import cv2
import numpy as np
 
img = cv2.imread('C_edges.jpeg')
print(img.shape) # Print image shape
cv2.imshow("original", img)

shape = img.shape

print(shape[0] , " " , shape[1])
    
h1 = int(input("enter pixels from top : "))
h2 = shape[0] - int(input("enter pixels from bottom : "))
x1 = int(input("enter pixels from left : "))
x2 = shape[1] - int(input("enter pixels from right : "))

print(type(h1))

# Cropping an image
cropped_image = img[h1:h2, x1:x2]

#                  h1   h2      x1    x2
 
# Display cropped image
cv2.imshow("cropped", cropped_image)
 
# Save the cropped image
cv2.imwrite("edges.jpeg", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()





















#Load image, grayscale, Gaussian blur, Otsu's threshold, dilate
import cv2
import numpy as np
from PIL import Image as im

name = []

# Load image, grayscale, Gaussian blur, Otsu's threshold, dilate
image = cv2.imread('edges.jpeg')

original = image.copy()

image = ~image
cv2.imwrite("img_inv.png",image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
#cv2.imshow('Gaussian',blur)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('Otsu',thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
dilate = cv2.dilate(thresh, kernel, iterations=10)

# Find contours, obtain bounding box coordinates, and extract ROI
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]
    
    cv2.imwrite("LOI_{}.png".format(image_number), ROI)
    name.append("LOI_{}.png".format(image_number))
    
    image_number += 1

#cv2.imshow('image', image)
#cv2.imshow('thresh', thresh)
#cv2.imshow('dilate', dilate)
print(dilate)
  # creating image object of
    # above array
data = im.fromarray(dilate)
      
    # saving the final output 
    # as a PNG file
data.save('outh.png')
#dilate.save('test_gray2.jpg')
cv2.waitKey()

print(name)




# In[4]:


import numpy as np

#digit = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype='f')
digit = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

pathlist = ['0.png','1.png','2.png','3.png','4.png','5.png','6.png','7.png','8.png','9.png' ]

newpathlist = ['0_grayImg.png','1_grayImg.png','2_grayImg.png','3_grayImg.png','4_grayImg.png','5_grayImg.png',
               '6_grayImg.png','7_grayImg.png','8_grayImg.png','9_grayImg.png' ]

savelist = ['0_gray.png', '1_gray.png', '2_gray.png', '3_gray.png', '4_gray.png' , '5_gray.png', '6_gray.png',
           '7_gray.png', '8_gray.png', '9_gray.png']


digits_value = [[[]]]


import cv2

from PIL import Image
# RGB to Gray image


def Running_Algo():

    for index_No in range(10):
        #print(index_No)
        img = Image.open(pathlist[index_No])
        imgGray = img.convert('L')
        imgGray.save(newpathlist[index_No])

        image = Image.open(newpathlist[index_No])
        new_image = image.resize((28, 28))
        new_image.save(savelist[index_No])
        import cv2
        import numpy as np
        from PIL import Image as im
        import matplotlib.pyplot as plt

        image = cv2.imread(savelist[index_No])
        plt.imshow(image, cmap="binary")
        #print(image.shape)

        original = image.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = np.reshape(gray, (-1, 784))
        gray = gray.astype('float32') / 255
        zero_base = gray.copy()
        zero_base.shape

        digits_value.append(zero_base)

    else:
        print("Finally finished!")
    #print(zero_base)
    
Running_Algo()


# In[5]:


import cv2

from PIL import Image
# RGB to Gray image


def reconization_Digit(finding_Digit_Path):

    img = Image.open(finding_Digit_Path)
    imgGray = img.convert('L')
    imgGray.save(saving_Digit_image)

    image = Image.open(saving_Digit_image)
    new_image = image.resize((28, 28))
    new_image.save(saving_Digit_image)
    import cv2
    import numpy as np
    from PIL import Image as im
    import matplotlib.pyplot as plt

    image = cv2.imread(saving_Digit_image)
    plt.imshow(image, cmap="binary")
    #print(image.shape)

    original = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.reshape(gray, (-1, 784))
    gray = gray.astype('float32') / 255
    finding_digit = gray.copy()
    return finding_digit
    


#finding_Digit_Path = 'LOI_1.png'

saving_Digit_image = 'ROI_70_gray.png'




#finding_digit = reconization_Digit(finding_Digit_Path)


# In[6]:


def finding_Distance():
    for i in range (10):
        temp = finding_digit[0] - digits_value[i+1][0]
        distance = np.sqrt(np.dot(temp.T, temp))
        digit[i] =  distance

        print("Euclidean Distance: ", distance)


# In[7]:


total = 0
size = len(name)
for i in range(size):
    finding_digit = reconization_Digit(name[i])
    finding_Distance()
    total = total + digit.index(min(digit))
    print( digit.index(min(digit)) ,"  +  ", )
    #print(i)

print(total)


# In[14]:


print(total)


# In[ ]:





# In[17]:





# In[ ]:




