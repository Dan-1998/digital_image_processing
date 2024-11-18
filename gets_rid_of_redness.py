import cv2
import numpy as np

wb = cv2.xphoto.createSimpleWB()
    
def read_image(img_path):
    return cv2.imread(img_path)

def write_image(img_path, img):
    cv2.imwrite(img_path, img)

def normalize(pixel):
    return np.float32(pixel) / 255

def unnormalize(pixel):
    return np.uint8(pixel * 255)

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

def white_balanced_image(img, alpha=2.5, compensate_blue_channel=False):
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

    img = gray_world(img)#wb.balanceWhite(img)
    return img

def underwater_image_enhancement():
    pass

def main():
    img = read_image('input_image1.png')
    img1 = white_balanced_image(img)
    write_image("original_image_modified.png", img)
    img2 = white_balanced_image(img, compensate_blue_channel=True)
    write_image("help_again.png", img1)
    write_image("help_again_again.png", img2)
    
if __name__ == '__main__':
    main()
