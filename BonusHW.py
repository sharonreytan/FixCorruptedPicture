# details = ("307923151","Sharon","Reytan")
# partner = ("208455600","Ron","Katz")


import copy
import numpy as np
import cv2


# check if an image is in grayscale
def is_grayscale(img):
    if img.ndim == 2:
        return 1
    else:
        return 0


# a function that makes every pixel equal to the median of its surroundings
def bw_mean_filter(mat):
    wid, hei = mat.shape
    # clone the given mat, to keep the values of the old mat, for best results
    mean_mat = copy.copy(mat)
    # pad the original mat with 1, so we can calculate all of it
    padded_mat = cv2.copyMakeBorder(mean_mat, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    for i in range(0, wid):
        for j in range(0, hei):
            # for each pixel - calculate the pixel's neighbours mean
            cell_mean_of_neib = np.mean(padded_mat[i:i + 3, j:j + 3])
            # make this cell black - it is the background (closer to a white surface)
            if cell_mean_of_neib > 127:
                mean_mat[i][j] = 255
            # make this cell white - it is the object (closer to a black surface)
            else:
                mean_mat[i][j] = 0
    return mean_mat


# a function that makes every white cell purple, and every black cell green
def bw_to_purple_green(bw_img):
    # in order to get purple, we need blue and red to be at 127 where the image is white (e.g. the pixel = 0)
    # for this purpose, we apply bitwise-not for each pixel. this way every 0 wil become 255.
    bw_img_reverted = ~bw_img
    return cv2.merge((bw_img // 2, bw_img_reverted, bw_img // 2))


def ObjectColoring(Img):
    # check if the image is grayscale
    if is_grayscale(Img):
        # make the image black & white and reduce noise
        img_bw = bw_mean_filter(Img)
    else:
        img_grayscale = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        img_bw = bw_mean_filter(img_grayscale)
    # make white appear purple, and black appear green
    img_purple_green = bw_to_purple_green(img_bw)
    return img_purple_green



