
# Part 2: Edge Detection

import cv2
import matplotlib.pyplot as plt

# Question 1: Convert image to grayscale
image = cv2.imread('Credit.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.show()

# Question 2: Apply Sobel filter
sobel_x = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_8U, 0, 1, ksize=3)
sobel_xy = cv2.Sobel(gray_image, cv2.CV_8U, 1, 1, ksize=3)

plt.imshow(sobel_x, cmap='gray')
plt.title("Sobel dx=1, dy=0")
plt.show()

plt.imshow(sobel_y, cmap='gray')
plt.title("Sobel dx=0, dy=1")
plt.show()

plt.imshow(sobel_xy, cmap='gray')
plt.title("Sobel dx=1, dy=1")
plt.show()

# Question 3: Compare with Canny edge detector
canny_1 = cv2.Canny(gray_image, 100, 200)
canny_2 = cv2.Canny(gray_image, 50, 150)

plt.imshow(canny_1, cmap='gray')
plt.title("Canny (100, 200)")
plt.show()

plt.imshow(canny_2, cmap='gray')
plt.title("Canny (50, 150)")
plt.show()
