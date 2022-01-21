import cv2
import imutils

image = cv2.imread("Images/Image2.jpg")
image = cv2.resize(image, (1270, 353))
cv2.imshow("Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray, (7, 7), 0)
#gray = cv2.bilateralFilter(image, 15,75,75)

# apply binary thresholding
ret, thresh = cv2.threshold(gray, 50, 100, cv2.THRESH_BINARY)

# visualize the binary image
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
print(contours)
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
tltrX = 180
tltrY = 300

blbrX = 235
blbrY = 300

cv2.line(image_copy, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(0, 255, 0), 2)
# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)