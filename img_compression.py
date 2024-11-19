import cv2
import numpy as np
import random
import math

class Pixel:
    def __init__(self, b, g, r):
        self.b = b
        self.g = g
        self.r = r

def euclidean_distance(x1, y1, c1, x2, y2, c2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (c1 - c2) ** 2)

def random_int(lim):
    return random.randint(0, lim)

def new_cluster_centers(image, cluster_centres, labels, number_of_colors):
    rw, cl, _ = image.shape
    for i in range(rw):
        for j in range(cl):
            b1, g1, r1 = image[i, j]
            mindistance = float('inf')
            centroid_label = 0
            for t in range(number_of_colors):
                val1, val2, val3 = cluster_centres[t].b, cluster_centres[t].g, cluster_centres[t].r
                distance = euclidean_distance(val1, val2, val3, b1, g1, r1)
                if distance < mindistance:
                    mindistance = distance
                    centroid_label = t
            labels[i, j] = centroid_label

def find_centroids(image, labels, cluster_centres, number_of_colors):
    rw, cl, _ = image.shape
    for i in range(number_of_colors):
        val1 = val2 = val3 = 0
        it = 0
        for j in range(rw):
            for k in range(cl):
                if labels[j, k] == i:
                    b, g, r = image[j, k]
                    val1 += b
                    val2 += g
                    val3 += r
                    it += 1
        if it > 0:
            cluster_centres[i] = Pixel(val1 // it, val2 // it, val3 // it)

def train_model(image, cluster_centres, labels, number_of_colors, steps):
    print("Image Compression Started")
    new_cluster_centers(image, cluster_centres, labels, number_of_colors)
    for step in range(steps):
        find_centroids(image, labels, cluster_centres, number_of_colors)
        new_cluster_centers(image, cluster_centres, labels, number_of_colors)
        print(f"Working on Compression Step: {step + 1}")

# Get user input
input_image = input("Enter the input image address: ")
number_of_colors = int(input("Enter the Number of colors you want to take in image: "))
output_image = "compressed_image.png"

# Validate steps input
steps = 61
while steps < 20 or steps == 61:
    steps = int(input("Enter the number of steps you want to train the model: "))
    if steps == 61:
        break
    elif steps < 10:
        print("Note: Steps Entered must be higher for better quality of image")
        print("Re-enter the number of steps you want to train the model: ")

# Load image
image = cv2.imread(input_image)

# JPEG compression for comparison
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 35]
_, buffer = cv2.imencode(".jpg", image, encode_param)
with open("jpeg_compressed.jpg", "wb") as f:
    f.write(buffer)

# Check if image is loaded properly
if image is None:
    print("Incorrect image address")
else:
    print("Image uploaded successfully")

    img_rows, img_cols, _ = image.shape
    labels = np.zeros((img_rows, img_cols), dtype=np.uint8)

# Initialize cluster centers with random pixels
    cluster_centres = []
    for _ in range(number_of_colors):
        random_r = random_int(img_rows - 1)
        random_c = random_int(img_cols - 1)
        b, g, r = image[random_r, random_c]
        cluster_centres.append(Pixel(b, g, r))

    # Train the model
    train_model(image, cluster_centres, labels, number_of_colors, steps)

    # Assign pixels to new cluster centers
    for i in range(img_rows):
        for j in range(img_cols):
            pixel = cluster_centres[labels[i, j]]
            image[i, j] = [pixel.b, pixel.g, pixel.r]

    # Write the compressed image
    cv2.imwrite(output_image, image)

    # Display the compressed image
    cv2.imshow("Display Our Model Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
