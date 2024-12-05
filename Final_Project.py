#Load libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

def read_image(img_path):
    return cv2.imread(img_path)

def write_image(img_path, img):
    cv2.imwrite(img_path, img)

def normalize(pixel):
    return np.float32(pixel) / 255

def unnormalize(pixel):
    return np.uint8(pixel * 255)

def convert(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
def gray_world(img):
    # Convert image to float32 for precision during calculations
    img_float = np.float32(img)

    # Calculate the average of each channel
    avg_b = np.mean(img_float[:, :, 0])  # Blue channel average
    avg_g = np.mean(img_float[:, :, 1])  # Green channel average
    avg_r = np.mean(img_float[:, :, 2])  # Red channel average

    # Calculate the overall average (mean of the three channels)
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    # Calculate scale factors for each channel
    scale_b = avg_all / avg_b
    scale_g = avg_all / avg_g
    scale_r = avg_all / avg_r

    # Apply the scaling factors to each channel
    img_float[:, :, 0] *= scale_b  # Scale Blue channel
    img_float[:, :, 1] *= scale_g  # Scale Green channel
    img_float[:, :, 2] *= scale_r  # Scale Red channel

    # Clip values to stay within valid range (0 to 255)
    img_float = np.clip(img_float, 0, 255)

    # Convert the image back to uint8
    img_corrected = np.uint8(img_float)

    return img_corrected
    
def white_balanced_image(img, alpha=1, compensate_blue_channel=False):
    """
    Apply white balancing to red channel of BGR underwater images
    
    Args:
        img (np.array): the image to apply white balancing to
        alpha (int) (optional): The weight to apply to color correction (default = 1)
        compensate_blue_channel (bool) (optional): Whether to also apply blue channel correction (need for cases of turbid waters, plankton) (default = False)
    """
    if len(img.shape) < 3:
        raise ValueError("Color image expected, received grayscale image")
    mean_red = np.mean(img[:, :, 2])
    mean_green = np.mean(img[:, :, 1])
    mean_blue = np.mean(img[:, :, 0]) if compensate_blue_channel else None
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j][2] = unnormalize(normalize(img[i][j][2]) + alpha * (normalize(mean_green) - normalize(mean_red)) * (1 - normalize(img[i][j][2])) * normalize(img[i][j][1]))
            if compensate_blue_channel:
                img[i][j][0] = unnormalize(normalize(img[i][j][0]) + alpha * (normalize(mean_green) - normalize(mean_blue)) * (1 - normalize(img[i][j][0])) * normalize(img[i][j][1]))
    return img
        
def normalized_unsharp_masking(img, kernel_size=7):
    b, g, r = cv2.split(img)
    sharpened_channels = []
    for channel in [b, g, r]:
        blurred_channel = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
        sharpened_channel = channel - blurred_channel
        contrast_stretched_channel = cv2.normalize(sharpened_channel, None, 0, 255, cv2.NORM_MINMAX)
        sharpened_channels.append(contrast_stretched_channel)#cv2.addWeighted(channel, 1 + weight, sharpened_img, weight, 0))
    mask = cv2.merge(sharpened_channels)
    return np.uint8((img + mask) / 2)

def gamma_correction(img, gamma_factor=1.2):
    return np.uint8((img/255)**gamma_factor * 255)

def laplacian_contrast_weight(img):
    return cv2.convertScaleAbs(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F))
    
def saliency_weight(img):
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    # Compute the mean pixel vector (I_mu)
    mean_pixel = lab_image.mean(axis=(0, 1), keepdims=True)  # Broadcast to full image size
    
    # Apply Gaussian blur to the original image (I_ωhc)
    blurred_image = cv2.GaussianBlur(lab_image, (3, 3), 0)
    
    # Compute the L2 norm (Euclidean distance) between I_mu and I_ωhc
    saliency_weight = np.linalg.norm(mean_pixel - blurred_image, axis=2)  # Across the Lab channels
    return saliency_weight

def saturation_weight(grayscale_img, color_img):
    grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #return np.linalg.norm(color_img - grayscale_img[..., np.newaxis], axis=2) / np.sqrt(3)
    return np.sqrt((color_img[:,:,0] - grayscale_img)**2 + (color_img[:,:,1] - grayscale_img)**2 + (color_img[:,:,2] - grayscale_img)**2) / np.sqrt(3)

def saturation_weight(img):
    b,g,r=cv2.split(img)
    lum=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    w=np.sqrt((1/3)*((r-lum)**2+(g-lum)**2+(b-lum)**2))
    return w
    
def combine_weights(img1mask1, img1mask2, img1mask3, img2mask1, img2mask2, img2mask3):
    img1_weight = img1mask1 + img1mask2 + img1mask3
    img2_weight = img2mask1 + img2mask2 + img2mask3
    img1_norm = (img1_weight + 0.1)/(img1_weight + img2_weight)
    img2_norm = (img2_weight + 0.2)/(img1_weight + img2_weight)
    img1_norm=np.repeat(img1_norm[:, :, np.newaxis], 3, axis=-1)
    img2_norm=np.repeat(img2_norm[:, :, np.newaxis], 3, axis=-1)
    return img1_norm, img2_norm

def gaussian_pyramid(image, levels):
    pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def laplacian_pyramid(image, levels):
    gaussian_pyramid_list = gaussian_pyramid(image, levels)
    laplacian_pyramid = [gaussian_pyramid_list[-1]]

    for i in range(levels - 1, 0, -1):
        size = (gaussian_pyramid_list[i - 1].shape[1], gaussian_pyramid_list[i - 1].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid_list[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid_list[i - 1], expanded)
        laplacian_pyramid.insert(0, laplacian)
    return laplacian_pyramid

def reconstruct_from_laplacian(pyramid):
    level = len(pyramid)

    for i in range(level - 1, 0, -1):
        m, n, c  = pyramid[i - 1].shape
        pyramid[i - 1] += pyramid[i - 1] + cv2.resize(pyramid[i], (n, m))
    output = pyramid[0]
    output=np.clip(output, 0, 255).astype(np.uint8)
    return output

def fusion(level, img_gamma_corrected, img_sharpened, combined_weight_gamma_corrected, combined_weight_sharpened):
    input1_laplacian_pyramid=laplacian_pyramid(img_gamma_corrected,level)
    input2_laplacian_pyramid=laplacian_pyramid(img_sharpened,level)

    combined_weight_gamma_corrected = gaussian_pyramid(combined_weight_gamma_corrected, level)
    combined_weight_sharpened = gaussian_pyramid(combined_weight_sharpened, level)

    result = []
    for i in range(level):
        fuse=np.multiply(combined_weight_gamma_corrected[i],input1_laplacian_pyramid[i]) + np.multiply(combined_weight_sharpened[i],input2_laplacian_pyramid[i])
        result.append(fuse)
    result = reconstruct_from_laplacian(result).astype(np.uint8)
    return result
    
def underwater_image_enhancement(img, title, compensate_blue_channel, gamma_factor, kernel_size, laplacian_levels):
    img_normalized = normalize(img.copy())
    img_balanced = white_balanced_image(img.copy(), compensate_blue_channel=True)
    write_image(f"{title}_white_balanced.jpg", img_balanced)
    
    img_grayworld = gray_world(img_balanced.copy())
    write_image(f"{title}_grayworld.jpg", img_grayworld)
    
    img_sharpened = normalized_unsharp_masking(img_grayworld.copy())
    write_image(f"{title}_sharpened.jpg", img_sharpened)
    
    img_gamma_corrected = gamma_correction(img_grayworld, gamma_factor=gamma_factor)
    write_image(f"{title}_gamma_corrected.jpg", img_gamma_corrected)
    
    grayscale_gamma_corrected = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_BGR2GRAY).astype(np.float64)/255
    # write_image(f"{title}_grayscale_gamma_corrected.jpg", grayscale_gamma_corrected) 
    
    grayscale_sharpened = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2GRAY).astype(np.float64)/255
    # write_image(f"{title}_grayscale_sharpened.jpg", grayscale_sharpened)
    
    laplacian_contrast_weight_gamma_corrected = laplacian_contrast_weight(img_gamma_corrected)#grayscale_gamma_corrected)
    # write_image(f"{title}_laplacian_contrast_gamma_corrected.jpg", laplacian_contrast_weight_gamma_corrected)
    
    laplacian_contrast_weight_sharpened = laplacian_contrast_weight(img_sharpened)#grayscale_sharpened) 
    # write_image(f"{title}_laplacian_contrast_sharpened.jpg", laplacian_contrast_weight_sharpened)

    saliency_weight_gamma_corrected = saliency_weight(img_gamma_corrected)
    write_image(f"{title}_saliency_weight_gamma_corrected.jpg", saliency_weight_gamma_corrected)  
    
    saliency_weight_sharpened = saliency_weight(img_sharpened)  
    write_image(f"{title}_saliency_weight_sharpened.jpg", saliency_weight_sharpened) 
    
    saturation_weight_gamma_corrected = saturation_weight(img_gamma_corrected)#grayscale_gamma_corrected, img_normalized)
    write_image(f"{title}_saturation_weight_gamma_corrected.jpg", unnormalize(saturation_weight_gamma_corrected)) 

    saturation_weight_sharpened = saturation_weight(img_sharpened)#grayscale_sharpened, img_normalized)  
    write_image(f"{title}_saturation_weight_sharpened.jpg", unnormalize(saturation_weight_sharpened)) 

    combined_weight_gamma_corrected, combined_weight_sharpened = combine_weights(laplacian_contrast_weight_gamma_corrected, saliency_weight_gamma_corrected, saturation_weight_gamma_corrected,
                                                                                 laplacian_contrast_weight_sharpened, saliency_weight_sharpened, saturation_weight_sharpened)
    write_image(f"{title}_combined_weight_gamma_corrected.jpg", unnormalize(combined_weight_gamma_corrected)) 
    write_image(f"{title}_combined_weight_sharpened.jpg", unnormalize(combined_weight_sharpened))

    result = fusion(laplacian_levels, img_gamma_corrected, img_sharpened, combined_weight_gamma_corrected, combined_weight_sharpened)

    cv2.imwrite(f'{title}_result.jpg', result)
    return result

def main():
    img_path = 'output/budda/budda_original.png'
    img = read_image(img_path)
    compensate_blue = True
    gamma_factor = 1.5
    kernel_size = 7
    laplacian_levels = 3
    underwater_image_enhancement(img, re.split('[_.]', img_path)[0], compensate_blue, gamma_factor, kernel_size, laplacian_levels)#img_path.split(['_', '.'])[0])
    print("Done")
if __name__ == '__main__':
    main()