# Name:
# Student:
# Number: 
# UPM Email : 


import cv2
import numpy as np
import skimage.io
import os
from skimage.filters import median , rank
import skimage.morphology as morphology

# function to add Gaussian noise 
def add_gaussian_noise(image, variance):
    """
    gaussian noise to the given image.
    parameters:
        image (numpy.ndarray): input image.
        variance (float): variance of the noise.
    returns:
        noisy_image (numpy.ndarray): noisy image.
    """
    noisy_img = image + np.random.normal(0, variance, image.shape)
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = noisy_img.astype(np.uint8)
    return noisy_img

# function for 2D convolution
def apply_2d_convolution(image, kernel, padding=0, strides=1):
    """
    applies 2D convolution on the input image.
    parameters:
        image (numpy.ndarray): Input image.
        kernel (numpy.ndarray): Convolution kernel.
        padding (int): Padding size.
        strides (int): Stride size.
    returns:
        output (numpy.ndarray): Convolved image.
    """
    # check if image is RGB and covert it to gray
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.flipud(np.fliplr(kernel))
    # take the x and y shape for image and kernel
    x_kernel_shape = kernel.shape[0]
    y_kernel_shape = kernel.shape[1]
    x_image_shape = image.shape[0]
    y_image_shape = image.shape[1]

    x_output = int(((x_image_shape - x_kernel_shape + 2 * padding) / strides) + 1)
    y_output = int(((y_image_shape - y_kernel_shape + 2 * padding) / strides) + 1)
    output = np.zeros((x_output, y_output))
    # apply padding 
    if padding != 0:
        image_padded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        image_padded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        image_padded = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - y_kernel_shape:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - x_kernel_shape:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * image_padded[x: x + x_kernel_shape, y: y + y_kernel_shape]).sum()
                except:
                    break

    return output

# edge detection kernel
edge_detection_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Function for loading X-ray images
def load_x_ray_images(folder_path):
    """
    loads X-ray images from a given folder.
    parameters:
        folder_path (str): path to the folder containing X-ray images.
    returns:
        x_ray_images (list): list of loaded X-ray images.
    """
    file_names = os.listdir(folder_path)
    image_files = [file for file in file_names if file.endswith('.png')]
    image_files.sort()
    x_ray_images = [skimage.io.imread(os.path.join(folder_path, file)) for file in image_files]
    return x_ray_images

# function for processing and saving convolution results
def process_and_save_convolution_results(x_ray_images, output_folder, noise_variance=0.01, num_stages=3):
    """
    processes X-ray images with noise reduction and convolution, and saves results.
    Parameters:
        x_ray_images (list): List of X-ray images to process.
        output_folder (str): Path to the folder where results will be saved.
        noise_variance (float): Variance of Gaussian noise to add (default is 0.01).
        num_stages (int): Number of convolution stages (default is 3).
    """
    for image_index, image in enumerate(x_ray_images):
        noisy_image = add_gaussian_noise(image, variance=noise_variance)
        denoised_image = rank.mean(noisy_image, morphology.disk(1))
        
        # create a subfolder for each X-ray image
        image_output_folder = os.path.join(output_folder, f'Image_{image_index + 1}')
        if not os.path.exists(image_output_folder):
            os.mkdir(image_output_folder)

        stage_output = denoised_image
        for stage in range(num_stages):
            stage_output = apply_2d_convolution(stage_output, edge_detection_kernel, padding=2)

            # save the output for each stage
            cv2.imwrite(os.path.join(image_output_folder, f'Stage{stage + 1}.jpg'), stage_output)

if __name__ == "__main__":
    # path to X-ray images
    folder_path = 'X-ray-dataset/'

    # load X-ray images
    x_ray_images = load_x_ray_images(folder_path)

    # folder for all stage results
    all_stage_results_folder = 'All_Stage_Results'

    # process and save convolution results
    process_and_save_convolution_results(x_ray_images, all_stage_results_folder, noise_variance=0.01, num_stages=3)
