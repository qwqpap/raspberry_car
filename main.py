import cv2
import numpy as np
import time
import pylab

video = cv2.VideoCapture("turn.mp4")


def calcGrayHist(img):
    # 计算灰度直方图
    i = 1
    j = 1
    k = 0
    fina_list = [0]
    while i <= 1719:
        while j <= 699:
            if img[j, i] < 254:
                pass
            elif img[j, i] >= 255:
                k = k + 1
            j = j + 1
        j = 0
        fina_list.append(k)
        k = 0
        i = i + 1

    #x = list(range(1720))

    #pylab.plot(x, fina_list)
    #pylab.show()
    js = 1
    way_value = 0
    for num in fina_list:
        way_value = num * js
        js = js + 1
    js = 1
    way_value = way_value/len(fina_list)
    print(way_value)
    return way_value


# 'camera' or 'picture'
gs = 9
mode = 'picture'


'''
这个是二值化的
'''
def line_finder(img=0):
    global video
    Hmin = 0
    Smin = 0
    Vmin = 0

    ret, img = video.read()
    img = img[200:900, 100:1820]

    ray_img = cv2.cvtColor(img, None, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(ray_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (gs, gs), 1, 2)
    gray_img = cv2.erode(gray_img, (3, 3), iterations=2)
    # hsv

    min_dff = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, -10)

    djf = min_dff

    qwq, djf = cv2.threshold(djf, 254, 255, cv2.THRESH_BINARY)
    #hsv 筛选
    color_dist = {'Lower': np.array([0, 11, 178]), 'Upper': np.array([23, 74, 223])}  # 这一坨是hsv 的参数 希望有苦力来调一下
    gs_frame = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 颜色空间转换-hsv
    erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀
    hsv_got = cv2.inRange(erode_hsv, color_dist['Lower'], color_dist['Upper'])

    #两个图层之合成
    hsv_got = ~hsv_got
    djf = ~djf
    finally_got = djf + hsv_got
    finally_got = ~finally_got
    cv2.imshow("origin_pic", img)
    cv2.imshow("djff", hsv_got)
    cv2.imshow("gs", finally_got)
    #inRange_hsv = cv2.inRange(erode_hsv, np.array([Hmin, Smin, Vmin]), np.array([Hmax, Smax, Vmax]))
    # end_time = time.time()

    # delta_time = end_time - star_time
    # fps = (1 / delta_time)
    # print(fps)
    #下面这个就是画直方图的，
    #calcGrayHist(djf)


    cv2.waitKey(20) & 0xff


def sewer_finder():
    global video
    ret, img = video.read()
    color_dist = {'Lower': np.array([0, 0, 0]), 'Upper': np.array([130, 255, 255])}  # 这一坨是hsv 的参数 希望有苦力来调一下
    img = img[300:900, 100:1820]
    gs_frame = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 颜色空间转换-hsv
    erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀
    fina_out = cv2.inRange(erode_hsv, color_dist['Lower'], color_dist['Upper'])  # hsv 抠图二值化
    cv2.imshow("gs", fina_out)
    cv2.waitKey(20) & 0xff
    '''
    以上是hsv抠图部分
    以下是拟合部分
    还没写完qwq
    '''

while True:
    # star_time = time.time()
    line_finder()
