import cv2
import numpy as np
import time


def process_image(image_path, precision):
    kernel = np.ones((3, 3), np.uint8)

    start = time.time()

    image0 = image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype('float') / 256.0

    image = cv2.dilate(image, kernel, iterations = 3)

    image = cv2.blur(image, (8, 8))
    ret, image = cv2.threshold(image, 1/64*15, 1, cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(np.uint8(image))

    labidx = 0
    for label in output[2]:
        if label[4] < 50000 / 900 * (precision**2):
            image[output[1] == labidx] = 0
        labidx += 1

    image = cv2.erode(image, kernel, iterations = 1)
    image[image0 == 0] = 0

    image = cv2.blur(image, (3, 3))
    ret, image = cv2.threshold(image, 1/9*3, 1, cv2.THRESH_BINARY)
    image[image0 < 0.01] = 0

    image = cv2.dilate(image, kernel, iterations = 3)
    image = cv2.GaussianBlur(image, (11, 11), 2)

    image = cv2.erode(image, kernel, iterations = 8)

    cv2.imwrite('edited-full__final.png', image * 256.0)

    print("Image elaborated in", time.time() - start, "s")


def add_glow(image_path):
    image = cv2.imread(image_path, 0)
    with_glow = cv2.GaussianBlur(image, (9, 9), 3)

    output = np.maximum(image, with_glow)

    cv2.imwrite("{}.glow.png".format(image_path), output)


def add_flooding(river_path, height_path, flooding_meters):
    river_image = cv2.imread(river_path, cv2.IMREAD_GRAYSCALE).astype('float') / 256.0
    height_image = cv2.imread(height_path, cv2.IMREAD_GRAYSCALE).astype('float') / 256.0

    flood_blur = cv2.GaussianBlur(river_image, (flooding_meters, flooding_meters), 0)
    # flood_blur = np.maximum(river_image, flood_blur)

    flood_hmask = height_image.copy()
    flood_hmask[river_image == 0] = 0

    kernel = np.ones((flooding_meters, flooding_meters), np.uint8)

    # y, x = np.ogrid[-flooding_meters/2:flooding_meters/2, -flooding_meters/2:flooding_meters/2]
    # mask = x * x + y * y <= (flooding_meters / 2) ** 2
    # kernel = np.zeros((flooding_meters, flooding_meters), np.uint8)
    # kernel[mask] = 1

    # flood_blur /= np.max(np.max(flood_blur))
    # cv2.imwrite("/tmp/gaussian.png", flood_blur * 255.0)
    # cv2.imshow("Gaussian", flood_blur)
    # cv2.waitKey(0)

    flood_hmask = cv2.dilate(flood_hmask, kernel)
    flood_hmask = cv2.GaussianBlur(flood_hmask, (31, 31), 3)

    flood_deltah = np.subtract(flood_hmask.copy(), height_image.copy())
    flood_deltah[flood_deltah < 0] = 0
    flood_deltah *= 10

    flood_factor = np.multiply(flood_deltah, flood_blur)
    flood_factor /= np.max(np.max(flood_factor))

    flood_factor = np.maximum(river_image, flood_factor)

    # ret, flood_factor = cv2.threshold(flood_factor, 0.05, 1, cv2.THRESH_BINARY)


    # cv2.imshow("Hmask", flood_hmask)
    # cv2.imshow("Himage", river_image)
    # cv2.imshow("Flood", flood_factor)
    # cv2.imshow("Flood2", flood_deltah2)
    # cv2.waitKey(0)
    return flood_factor

    # cv2.imwrite(river_image, cv2.IMREAD_GRAYSCALE).astype('float') / 256.0
    # cv2.imwrite(height_image, cv2.IMREAD_GRAYSCALE).astype('float') / 256.0


def overlap_flood(rgb_image, flood_map, lower, upper, color, ratio):
    flood_image = cv2.imread(flood_map, cv2.IMREAD_GRAYSCALE).astype('float') / 256.0

    ret, flood_factor_low = cv2.threshold(flood_image, lower, 1, cv2.THRESH_BINARY)
    ret, flood_factor_up = cv2.threshold(flood_image, upper, 1, cv2.THRESH_BINARY)

    rgb_overlay = rgb_image.copy()

    for i in range(0, 3):
        rgb_overlay[np.logical_and(flood_factor_low > 0, flood_factor_up == 0)] = np.uint8(color)#cv2.add(rgb_image[:,:,0], flood_image * 1024.0, dtype=cv2.CV_8U)

    # cv2.imshow("Himage", river_image)
    # cv2.imshow("Flood", flood_factor)
    # cv2.imshow("Flood2", flood_deltah2)

    rgb_image = cv2.addWeighted(rgb_image, ratio, rgb_overlay, 1.0 - ratio, 0.0)

    return rgb_image


if __name__ == "__main__":
    # image_path = 'subset_0_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_C2RCC_3_rhow_B3.png'
    # image_path = 'subset_0_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_C2RCC_3_rhow_B3.png'
    image_path = 'subset_1_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_C2RCC2_rhown_B3 (2).bmp'

    # process_image(image_path, 24)
    # add_glow(image_path)
    # for i in range(50, 2000, 50):
    image = add_flooding("edited-full__final.png",
                 "subset_1_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_elevation.bmp",
                 801)
    # cv2.imwrite(f"flooding-regions-gradual.png", image * 255.0)

    # for i in range(50, 2000, 50):
    # image = overlap_flood("subset_1_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_RGB.bmp",
    #               f"flooding-tmp.png")
    sys.exit(0)
    #     cv2.imwrite(f"flood_overlap{i}.png", image)
    rgb_path = "subset_1_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_RGB.bmp"
    image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

    image *= 0

    image = overlap_flood(image,
                  "flooding-regions-gradual.png", 0.25, 1.0, [0, 0, 255], 0.0)#6)

    image = overlap_flood(image,
        "flooding-regions-gradual.png", 0.1, 0.25, [0, 200, 255], 0.0)#7)

    image = overlap_flood(image,
        "flooding-regions-gradual.png", 0.0, 0.1, [180, 0, 0], 0.0)#6)


    # cv2.imshow("Img", image)
    print(image.shape)
    # cv2.waitKey(0)



    cv2.imwrite('flooding-final-mask.png', np.uint8(image))


    # image_path = '/home/andrea/flat-earthers/subset_1_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_C2RCC2_rhown_B3.bmp'
    # image_path = '/home/andrea/flat-earthers/subset_1_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_C2RCC2_rhown_B3 (1).bmp'
    # image_path = '/home/andrea/flat-earthers/subset_1_of_S2B_MSIL1C_20200213T101029_N0209_R022_T32TQR_20200213T122453_resampled_C2RCC2_rhown_B3 (2).bmp'
