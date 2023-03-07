import cv2
from PIL import Image
import numpy as np
from tqdm import *
import matplotlib.pyplot as plt
from math import sqrt, log10
from PIL import Image

def sum_of_abs_diff(pixel_vals_1, pixel_vals_2):
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1
    return np.sum(abs(pixel_vals_1 - pixel_vals_2))

def compare_blocks(y, x, block_left, right_array, block_size=5):
    x_min = max(0, x - SEARCH_BLOCK_SIZE)
    x_max = min(right_array.shape[1], x + SEARCH_BLOCK_SIZE)
    first = True
    min_sad = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size,
                                  x: x+block_size]
        sad = sum_of_abs_diff(block_left, block_right)
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)
    return min_index

def get_disparity_map():
    left_array = np.asarray(cv2.imread(img_name_left, cv2.IMREAD_GRAYSCALE))
    right_array = np.asarray(cv2.imread(img_name_right, cv2.IMREAD_GRAYSCALE))
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    if left_array.shape != right_array.shape:
        print("Images don't have the same shape.")
        return
    h, w = left_array.shape
    disparity_map = np.zeros((h, w))
    for y in tqdm(range(BLOCK_SIZE, h-BLOCK_SIZE)):
        for x in range(BLOCK_SIZE, w-BLOCK_SIZE):
            block_left = left_array[y:y + BLOCK_SIZE,
                                    x:x + BLOCK_SIZE]
            min_index = compare_blocks(y, x, block_left,
                                       right_array,
                                       block_size=BLOCK_SIZE)
            disparity_map[y, x] = abs(min_index[1] - x)
    # print(disparity_map)
    plt.imshow(disparity_map, cmap='binary', interpolation='nearest')
    plt.savefig(img_name_result)
    calculateResults(disparity_map)

def calculateResults(disparity):
    disp_name_left = 'img/' + dir_name + '/disp2.png'
    disp_name = dir_name + '_' + str(BLOCK_SIZE) + '_' + str(SEARCH_BLOCK_SIZE)
    disp_img_left = np.asarray(cv2.imread(
        disp_name_left, cv2.IMREAD_GRAYSCALE))

    file = open(file_result_name, 'a')

    file.write(disp_name + '\n')
    file.write(f'LIM_BAD_PIXELS = {LIM_BAD_PIXELS} \n')

    sum_img_px = calculate_sum_pixels(disp_img_left)

    ssd = calculate_ssd(disparity, disp_img_left)
    result = f'SSD: {ssd} \n'
    file.write(result)

    sad = calculate_sad(disparity, disp_img_left)
    perc_sad = 1 - (sad / sum_img_px)
    result = f'SAD: {int(sad)} -> {perc_sad:.2%} \n'
    file.write(result)

    psnrL = PSNR(disparity, disp_img_left)
    result = f'PSNR: {psnrL:,.2f} \n'
    file.write(result)

    rmseL = RMSE(disparity, disp_img_left)
    result = f'RMSE: {rmseL:,.2f} \n'
    file.write(result)
    
    bad_pixels = calculate_bad_pixels(disparity, LIM_BAD_PIXELS)
    result = f'BAD PIXELS: {bad_pixels:,.2f}% \n\n'
    file.write(result)

    file.close()

def calculate_ssd(img1, img2):
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)

def RMSE(img1, img2):
    mse = np.square(np.subtract(img1, img2)).mean()
    rmse = sqrt(mse)
    return rmse

def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_sum_pixels(img1):
    image1 = img1
    pixels1 = image1
    pixels_sum = 0
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            pixels_sum += pixels1[i, j]
    return pixels_sum

def calculate_sad(img1, img2):
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    pixels1 = img1
    pixels2 = img2
    sad = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            sad += abs(pixels1[i, j] - pixels2[i, j])
    return sad

def calculate_bad_pixels(disp, lim):
    bad_pixels = 0
    for i in range(len(disp)):
        for j in range(len(disp[i])):
            if abs(disp[i][j]) > lim:
                bad_pixels += 1
    total_pixels = len(disp) * len(disp[0])
    perc_bad_pixels = (bad_pixels / total_pixels) * 100
    return perc_bad_pixels

if __name__ == '__main__':
    DIR_LIST = ['cones', 'teddy']
    VALUES_BLOCK_SIZE = [3]
    VALUES_SEARCH_BLOCK_SIZE = [9]
    LIM_BAD_PIXELS = 5

    for dir in DIR_LIST:
        dir_name = dir
        img_name_left = 'img/' + dir_name + '/im2.png'
        img_name_right = 'img/' + dir_name + '/im6.png'
        file_result_name = 'img/results/results.txt'
        for block in VALUES_BLOCK_SIZE:
            for search in VALUES_SEARCH_BLOCK_SIZE:
                BLOCK_SIZE = block
                SEARCH_BLOCK_SIZE = search
                img_name_result = 'img/results/' + dir_name + '_' + \
                    str(BLOCK_SIZE) + '_' + str(SEARCH_BLOCK_SIZE) + '.png'
                print('Computing: ' + 'DIR = ' + dir + ', BLOCK_SIZE = ' +
                      str(BLOCK_SIZE) + ' and SEARCH_BLOCK_SIZE = ' + str(SEARCH_BLOCK_SIZE))
                get_disparity_map()
