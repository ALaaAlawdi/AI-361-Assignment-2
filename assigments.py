# Name:
# Student:
# Number:
# UPM Email:

# Import necessary libraries
import cv2
import numpy as np
import skimage.io
import os
from skimage.filters import median
import skimage.morphology as morphology


# function for 2D convolution
def convolve2D(image, kernel, padding=0, strides=1):
    """ 
    applies 2D convolution on the input image.
    parameters:
        image (numpy.ndarray): Input image.
        kernel (numpy.ndarray): Convolution kernel.
        padding (int): Padding size.
        strides (int): Stride size.

    returns:
        output (numpy.ndarray): convolved image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.flipud(np.fliplr(kernel))

    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - yKernShape:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

# function to introduce Gaussian noise to an image
def introduce_noise(image, var):
    """
    gaussian noise to the given image.

    parameters:
        image (numpy.ndarray): Input image.
        var (float): Variance of the noise.

    peturns:
        noisy_image (numpy.ndarray): Noisy image.
    """
    noisy_image = image + np.random.normal(0, var, image.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

# edge detection kernel
edge_detection_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# define the path to the X-ray images
folder_path = 'X-ray-dataset/'

# list all the files in the folder
file_names = os.listdir(folder_path)

# filter only the PNG image files
image_files = [file for file in file_names if file.endswith('.png')]

# # Sort the image files
# image_files.sort()

# read the images from the folder
x_ray_images = [skimage.io.imread(os.path.join(folder_path, file)) for file in image_files]

# lists to store convolution results
convolution_results = []

for image in x_ray_images:
    # introduce Gaussian noise
    noisy_image = introduce_noise(image, var=0.01)

    # apply median filtering to reduce noise
    denoised_image = median(noisy_image, morphology.disk(1))

    # convolution in three stages
    stage1_output = convolve2D(denoised_image, edge_detection_kernel, padding=2)
    stage2_output = convolve2D(stage1_output, edge_detection_kernel, padding=2)
    stage3_output = convolve2D(stage2_output, edge_detection_kernel, padding=2)

    # store the outputs for each stage
    convolution_results.append((stage1_output, stage2_output, stage3_output))

# create folders for each stage
for stage in range(1, 4):
    stage_folder = f'Stage{stage}_Results'
    if not os.path.exists(stage_folder):
        os.mkdir(stage_folder)

# save the convolution results
for i, (stage1, stage2, stage3) in enumerate(convolution_results):
    stage1_folder = f'Stage1_Results'
    stage2_folder = f'Stage2_Results'
    stage3_folder = f'Stage3_Results'

    cv2.imwrite(os.path.join(stage1_folder, f'2DConvolved_Stage1_{i + 1}.jpg'), stage1)
    cv2.imwrite(os.path.join(stage2_folder, f'2DConvolved_Stage2_{i + 1}.jpg'), stage2)
    cv2.imwrite(os.path.join(stage3_folder, f'2DConvolved_Stage3_{i + 1}.jpg'), stage3)
