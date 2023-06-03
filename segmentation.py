from matplotlib import pyplot as plt
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFilter
import warnings
import colorsys

warnings.filterwarnings("ignore")


def clustering(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #resize = int(img.shape[1] / 500)
    #width = int(img.shape[1] / resize)
    #height = int(img.shape[0] / resize)
    #dim = (width, height)
    #img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    shape = img.shape
    img = img.reshape((img.shape[1] * img.shape[0], 3))

    red_detector = 0
    k = 2
    while (red_detector == 0):
        k += 1
        print(f'\nКол-во кластеров = {k}')
        kmeans = KMeans(n_clusters=k)
        s = kmeans.fit(img)

        clusters = kmeans.cluster_centers_
        for i in range(len(clusters)):
            if (clusters[i][0] < 10 or clusters[i][0] > 170) and ((clusters[i][1] > 150) and (clusters[i][2] > 100) or (clusters[i][1] > 100) and (clusters[i][2] > 150)):
                red_detector = 1
                break
        # k += 1
        centroid = kmeans.cluster_centers_
        print('\nЦентры кластеров:\n', centroid)

    labels = kmeans.labels_
    labels = list(labels)

    percent = []
    for i in range(len(centroid)):
        j = labels.count(i)
        j = j / (len(labels))
        percent.append(j)

    #rgb_colors = cv2.cvtColor(np.array(centroid / 255), cv2.COLOR_HSV2RGB)
    rgb_color = []
    for i in range(len(centroid)):
        r,g,b = colorsys.hsv_to_rgb(centroid[i][0]/255,centroid[i][1]/255,centroid[i][2]/255)
        rgb_color.append([r,g,b])
    #print('Круговая диаграмма центров кластеров:')
    #plt.pie(percent, colors=rgb_color, labels=np.arange(len(centroid)))
    plt.pie(percent, colors=rgb_color, labels=np.around(centroid,0))
    plt.show()
    #print()

    centroid = np.uint8(centroid)
    res = centroid[(np.array(labels)).flatten()]
    result_image = res.reshape((shape))

    result_image = cv2.cvtColor(result_image, cv2.COLOR_HSV2RGB)
    # print('Итоговая кластеризация:')
    plt.imshow(result_image)
    plt.show()

    return kmeans, result_image


def thresh_bin(img_clust):
    img_hsv = cv2.cvtColor(img_clust, cv2.COLOR_RGB2HSV)

    heigth, width, channels = img_hsv.shape
    red_letters_img = np.zeros((heigth, width, channels))

    for i in range(heigth):
        for j in range(width):
            if (img_hsv[i][j][0] < 10 or img_hsv[i][j][0] > 170) and ((img_hsv[i][j][1] > 150 and img_hsv[i][j][2] > 100) or (img_hsv[i][j][1] > 100 and img_hsv[i][j][2] > 150)):
                red_letters_img[i][j][:] = 255
            else:
                red_letters_img[i][j][:] = 0

    red_letters_img = red_letters_img.astype(np.uint8)

    res_img = Image.fromarray(red_letters_img)
    #res_img = res_img.filter(ImageFilter.MedianFilter(size=3))
    #res_img = res_img.filter(ImageFilter.MedianFilter(size=3))
    #res_img = res_img.filter(ImageFilter.MaxFilter(size=7))
    plt.imshow(res_img)
    plt.show()

    return res_img


def draw_segmentation(bin_image,img):
    bin_image = np.asarray(bin_image)
    gray = cv2.cvtColor(bin_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((30, 30), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    #thresh = thresh.filter(ImageFilter.MaxFilter(size=10))

    plt.imshow(thresh)
    plt.show()

    result = img.copy()
    #thresh = 255 - thresh
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    #print(hierarchy)
    #print(contours)
    i = 0
    for cntr in contours:
        #if hierarchy[0,i,3] == -1:
        x, y, w, h = cv2.boundingRect(cntr)
        if w < 3 * h:
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)
        #i += 1

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.show()

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result


if __name__ == "__main__":
    imgpath = 'images/1.jpg'

    img = cv2.imread(imgpath)
    _, img_clust = clustering(img)
    draw_segmentation(thresh_bin(img_clust),img)
    cv2.waitKey()
