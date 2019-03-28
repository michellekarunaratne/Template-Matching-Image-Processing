import  cv2
from matplotlib import pyplot as plt
import numpy as np


#reading images
imageGrey=cv2.imread('page.png',0)
templateGrey=cv2.imread('letter.png',0)

#template matching
result=cv2.matchTemplate(imageGrey,templateGrey,cv2.TM_SQDIFF_NORMED)

#finding the occurences of the template
threshold=0.0135
templateHeight,templateWidth= templateGrey.shape

#checking the normalized squared difference value is smaller than threshold value so thats its closer to the template or equal to the template
location=np.where(result <= threshold)

#drawing the rectangle
for point in zip(*location[::-1]):
    cv2.rectangle(imageGrey, point,(point[0]+templateWidth,point[1]+templateHeight),0,2)

#Displaying the image
cv2.imshow("Detected Template",imageGrey)
cv2.imshow("Template",templateGrey)
cv2.waitKey(0)