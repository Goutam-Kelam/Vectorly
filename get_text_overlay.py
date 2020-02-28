# This is the problem for First technical round for the role of Computer Vision Engineer at Vectorly
# More details at https://www.linkedin.com/jobs/view/1629909785/
#
# Write a function which will segment and extract the text overlay "Bart & Homer's EXCELLENT Adventure" 
# Input image is at https://vectorly.io/demo/simpsons_frame0.png
# Output : Image with only the overlay visible and everything else white
# 
# Note that you don't need to extract the text, the output is an image with only 
# the overlay visible and everything else (background) white
#
# You can use the snipped below (in python) to get started if you like 
# Python is not required but is preferred. You are free to use any libraries or any language


#####################
import cv2
import numpy as np

def getTextOverlay(input_image):
    # Write your code here for output
    gray_img = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

    # Threshold the image 
    _, thresh_img = cv2.threshold(gray_img,7,255,cv2.THRESH_BINARY_INV)

    # The thresholded image contains small unwanted blobs, we need to remove them
    _, contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # loop over the contours and create a mask for the unwanted blobs
    mask = np.ones(gray_img.shape, dtype="uint8") * 255
    for cntr in contours:
        if cv2.contourArea(cntr)<500:
            cv2.drawContours(mask, [cntr], -1, 0, -1)
    
    # Remove the unwanted blobs from the thresholded image    
    thresh_img = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)

    # In order to remove unwanted blobs inside the letter "B" 
    # we used cv2.RETR_LIST to find all contours, and then removed contour whose area was less than 500
    # This leads to creation of holes in the text, which we fix my morphological operations.
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    thresh_img = cv2.morphologyEx(thresh_img,cv2.MORPH_CLOSE,kernel)

    # Create a smooth mask for blending foreground and background
    mask = cv2.dilate(thresh_img, None, iterations=10)
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.GaussianBlur(thresh_img, (3,3), 0)
    mask = mask.astype(float)/255

    # Create foreground and background image for blending
    background = np.ones(gray_img.shape, dtype="float") * 255
    foreground_img = cv2.multiply(mask, gray_img.astype(float))
    background_img = cv2.multiply(1-mask, background)

    # Create the overlay image and normalize it
    overlay_img = cv2.add(foreground_img, background_img)

    output = overlay_img
    
    return output

if __name__ == '__main__':
    image = cv2.imread('simpsons_frame0.png')
    output = getTextOverlay(image)
    cv2.imwrite('simpons_text.png', output)
#####################

