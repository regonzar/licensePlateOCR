import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io, img_as_uint
# from skimage.transform import rescale
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.filters import try_all_threshold, threshold_otsu

# file = './camion3.png'

# image = color.rgb2gray(io.imread(file))
# image = rescale(image, 5.0, mode='reflect')
# image = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15, multichannel=False)
# thresh = threshold_otsu(image)
# image = img_as_uint(image > thresh)

# tol = 0
# mask = image > 0
# image = image[np.ix_(mask.any(1), mask.any(0))]

# fig, ax = try_all_threshold(image, figsize=(10, 8), verbose=False)

# io.imsave('./license3processed.png', image)

# fig, ax = plt.subplots()

# ax.imshow(image, cmap='gray')

# plt.show()


def processImage(file='./camion3.png', output='./licence3processed.png',
                 mask_tol=0):

    image = color.rgb2gray(io.imread(file))
    # image = rescale(image, 5.0, mode='reflect')
    image = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15, multichannel=False)
    thresh = threshold_otsu(image)
    image = img_as_uint(image > thresh)

    mask = image > mask_tol
    image = image[np.ix_(mask.any(1), mask.any(0))]

    io.imsave('./license3processed.png', image)
