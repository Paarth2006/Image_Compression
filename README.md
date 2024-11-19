# Image_Compression using K-means Clustering
Overview
This project implements an image compression technique using K-means clustering, a type of unsupervised machine learning algorithm. The goal of the project is to reduce the number of colors in an image while maintaining as much of the visual quality as possible. The algorithm uses K-means clustering to group similar pixels into clusters, with each cluster being represented by its centroid. This helps in reducing the file size of images while preserving the overall appearance.

Features
Compresses an image by reducing the number of colors (using K-means clustering).
Supports user-defined number of colors (clusters) for compression.
Allows adjustable number of training steps to balance compression quality and performance.
Provides a comparison of the compressed image with JPEG compression.
Outputs the compressed image as PNG format.
Prerequisites
Python 3.x
OpenCV library (cv2)
NumPy library (numpy)

How It Works?
The script performs the following steps:

1> User Input:
  The user is prompted to provide an image file path and the number of color clusters they want to reduce the image to.
  The user also defines how many training steps the model should perform for better quality of compression.

2> Image Loading:
  The image is loaded into memory using OpenCV's cv2.imread() function.

3> Initial Cluster Centers:
  Random pixels from the image are selected as initial cluster centers. The number of clusters corresponds to the number of colors the user wants the image to be compressed to.

4> K-means Algorithm:
  The algorithm then proceeds in a loop for a defined number of training steps. In each step, the following happens:
  Cluster Assignment: Each pixel is assigned to the nearest cluster center (based on Euclidean distance between pixel colors and cluster centroids).
  Centroid Calculation: New centroids are computed by averaging the pixels assigned to each cluster.
  Repeat: This process is repeated for the given number of steps to refine the clustering.

5> Compression:
  After training, each pixel is replaced with its corresponding cluster's centroid value.
  The final compressed image is saved in PNG format and displayed to the user.

6> JPEG Compression Comparison:
  A JPEG compressed version of the original image is also created and saved as jpeg_compressed.jpg for comparison with the K-means compressed image.
