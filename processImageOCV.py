import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
np.set_printoptions(threshold=np.inf)


def processImage(file='./camion3.png', output='./license3processed.png',
                 mask_tol=0):

    img = cv2.imread(file)
    img = cv2.resize(img, (0, 0), fx=5, fy=5)

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    dns = cv2.fastNlMeansDenoising(gray_img, None, 10)

    ret, thr = cv2.threshold(dns, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = thr > mask_tol
    image = thr[np.ix_(mask.any(1), mask.any(0))]

    # Check if pre-filter is needed

    hist, bins = np.histogram(image, bins=[0, 122, 255])
    hist = np.array(hist).astype(float)
    white_perc = hist[1] / (hist[0] + hist[1])

    if white_perc > 0.75:
        image = preFilter(image)

    image = firstFilter(image)
    image = secondFilter(image)
    image = getBoundingBoxes(image)

    cv2.imwrite(output, image)


def preFilter(img):

    print('Pre-Filter entered')

    row_mean = np.mean(img, axis=1)
    col_mean = np.mean(img, axis=0)


    for i, row in enumerate(row_mean):
        hist, bins = np.histogram(img[i, :], bins=[0, 122, 255])
        hist = np.array(hist).astype(float)

        if (hist[0] / (hist[0] + hist[1]) > 0.1):
            mask_row_upper = i
            break

    for i in reversed(range(len(row_mean))):
        hist, bins = np.histogram(img[i, :], bins=[0, 122, 255])
        hist = np.array(hist).astype(float)

        if (hist[0] / (hist[0] + hist[1]) > 0.1):
            mask_row_lower = i
            break

    for i, col in enumerate(col_mean):
        hist, bins = np.histogram(img[:, i], bins=[0, 122, 255])
        hist = np.array(hist).astype(float)

        if (hist[0] / (hist[0] + hist[1]) > 0.25):
            mask_col_left = i
            break

    for i in reversed(range(len(col_mean))):
        hist, bins = np.histogram(img[:, i], bins=[0, 122, 255])
        hist = np.array(hist).astype(float)

        if (hist[0] / (hist[0] + hist[1]) > 0.25):
            mask_col_right = i
            break

    img = img[
        np.ix_([x for x in range(mask_row_upper, mask_row_lower + 1)],
               [x for x in range(mask_col_left, mask_col_right + 1)])
    ]

    # plt.subplot()
    # plt.imshow(img)
    # plt.show()

    return img


def firstFilter(img):

    row_mean = np.mean(img, axis=1)
    col_mean = np.mean(img, axis=0)

    for i, row in enumerate(row_mean):
        hist, bins = np.histogram(img[i, :], bins=[0, 122, 255])
        hist = np.array(hist).astype(float)

        if (hist[1] / (hist[0] + hist[1]) > 0.551):
            mask_row_upper = i
            break

    for i in reversed(range(len(row_mean))):
        hist, bins = np.histogram(img[i, :], bins=[0, 122, 255])
        hist = np.array(hist).astype(float)

        if (hist[1] / (hist[0] + hist[1]) > 0.76):
            mask_row_lower = i
            break

    for i, col in enumerate(col_mean):
        hist, bins = np.histogram(img[:, i], bins=[0, 122, 255])
        hist = np.array(hist).astype(float)

        if (hist[1] / (hist[0] + hist[1]) > 0.6):
            mask_col_left = i
            break

    for i in reversed(range(len(col_mean))):
        hist, bins = np.histogram(img[:, i], bins=[0, 122, 255])
        hist = np.array(hist).astype(float)

        if (hist[1] / (hist[0] + hist[1]) > 0.7):
            mask_col_right = i
            break

    img = img[
        np.ix_([x for x in range(mask_row_upper, mask_row_lower + 1)],
               [x for x in range(mask_col_left, mask_col_right + 1)])
    ]

    return img


def secondFilter(img):

    row_mean = np.mean(img, axis=1)

    for i, mean in enumerate(row_mean):
        if mean < 200:
            img[i, :] = 255
        else:
            break

    for i, mean in enumerate(reversed(row_mean)):
        if mean < 255:
            img[len(row_mean) - i - 1, :] = 255
        else:
            break

    col_mean = np.mean(img, axis=0)

    for i, mean in enumerate(col_mean):
        if mean < 200:
            img[:, i] = 255
        else:
            break

    for i, mean in enumerate(reversed(col_mean)):
        if mean < 200:
            img[:, len(col_mean) - i - 1] = 255
        else:
            break

    return img


def getBoundingBoxes(img):

    img[0, :] = 255
    img[-1, :] = 255
    img[:, 0] = 255
    img[:, -1] = 255

    image = copy.deepcopy(img)

    image, cnt, h = cv2.findContours(image, cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)

    w, h = img.shape[:2]
    box_mask = np.zeros((w, h), np.int8)

    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)

        if w*h > 8000 and w*h < 25000:
            box_mask = cv2.rectangle(box_mask, (x, y), (x+w, y+h), 255, -1)

    inv_img = cv2.bitwise_not(img)
    masked_data = cv2.bitwise_and(inv_img, inv_img, mask=box_mask)
    img = cv2.bitwise_not(masked_data)

    plt.subplot()
    plt.imshow(img)
    plt.show()
    return img
