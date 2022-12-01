import cv2
import numpy as np
import time

from numba import jit
#  直方图绘制所需
import matplotlib.pyplot as plt



left_start = 100
right_start = 1819
up_start = 400
down_start = 430
pressed_pix = 5
video = cv2.VideoCapture("turn1.mp4")

@jit
def find_value(img11):
    # 计算灰度直方图
    global left_start, right_start, up_start, down_start
    finally_list = []

    mid_num = int((right_start + left_start) / 2)
    for i in range(left_start, right_start):
        # error_now = i - mid_num
        n = 0
        for j in range(up_start, down_start):
            if img11[j, i] >= 128:
                n = n + 1
            else:
                pass
        finally_list.append(n)
    print(finally_list)
    '''
    x = np.array(range(0,len(finally_list)))
    y = np.array(finally_list)
    plt.bar(x,y,0.8)
    plt.show()
    '''
    left_line = left_start
    right_line = right_start
    # fina_error = int(fina_error/sigema + mid_num)
    for i in range(mid_num, right_start-left_start):
        if finally_list[i] >= pressed_pix:
            right_line = i + left_start
            print(i)
            break
        else:
            pass
    for i in range(mid_num, 0, -1):
        if finally_list[i] >= 1+pressed_pix:
            left_line = i + left_start
            print(i)
            break
        else:
            pass
    line = (left_line + right_line) / 2
    return line, left_line, right_line


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
    # img = img[200:900, 100:1820]

    # ray_img = cv2.cvtColor(img, None, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (gs, gs), 1, 2)
    gray_img = cv2.erode(gray_img, (3, 3), iterations=2)
    # hsv

    min_dff = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, -7)

    djf = min_dff

    qwq, djf = cv2.threshold(djf, 254, 255, cv2.THRESH_BINARY)
    # hsv 筛选
    color_dist = {'Lower': np.array([0, 11, 178]), 'Upper': np.array([23, 74, 223])}  # 这一坨是hsv 的参数 希望有苦力来调一下
    gs_frame = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 颜色空间转换-hsv
    erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀
    hsv_got = cv2.inRange(erode_hsv, color_dist['Lower'], color_dist['Upper'])

    # 两个图层之合成
    hsv_got = ~hsv_got
    djf = ~djf
    finally_got = djf + hsv_got
    finally_got = ~finally_got
    finally_got = cv2.dilate(finally_got, (5, 30), 5)
    line, left_line, right_line = find_value(finally_got)

    #print(line)
    cv2.rectangle(finally_got, (left_start, up_start), (right_start, down_start), (255, 255, 255), 3)
    cv2.rectangle(finally_got, (960, up_start), (960, down_start), (255, 255, 255), 3)
    cv2.rectangle(img, (left_line, up_start), (right_line, down_start), (255, 255, 255), 3)
    cv2.line(img,(int(line), int(down_start)), (int(line), int(up_start)),(255,255,255), 3)
    #cv2.rectangle(img, (line, up_start), (line, down_start), (255, 255, 255), 3)

    # 别管
    cv2.namedWindow("origin_Pic", 0);
    cv2.resizeWindow("origin_Pic", 1920, 1080);
    cv2.imshow("origin_Pic", img)
    cv2.namedWindow("gs", 0);
    cv2.resizeWindow("gs", 1920, 1080);
    cv2.imshow("gs", hsv_got)
    cv2.namedWindow("shit", 0);
    cv2.resizeWindow("shit", 1920, 1080);
    cv2.imshow("shit", finally_got)
    # inRange_hsv = cv2.inRange(erode_hsv, np.array([Hmin, Smin, Vmin]), np.array([Hmax, Smax, Vmax]))
    # end_time = time.time()

    # delta_time = end_time - star_time
    # fps = (1 / delta_time)
    # print(fps)
    # 下面这个就是画直方图的，
    # calcGrayHist(djf)

    cv2.waitKey(20) & 0xff


def sewer_finder():
    global video
    ret, img = video.read()
    color_dist = {'Lower': np.array([0, 0, 0]), 'Upper': np.array([130, 255, 255])}  # 这一坨是hsv 的参数 希望有苦力来调一下
    # img = img[300:900, 100:1820]
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
