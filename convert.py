#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
from PIL import Image
from math import pi,sin,cos,tan,atan2,asin,acos,hypot,floor,fabs,sqrt,radians
from numpy import clip,array,matrix

# convertCoordinate from coordinate A to B
# x 是沿所求坐标系转换为原坐标系X轴旋转的角度
# y 是沿所求坐标系转换为原坐标系y轴旋转的角度
# z 是沿所求坐标系转换为原坐标系z轴旋转的角度
# sequence 是矩阵旋转的次序，0是x->y->z，1是x->z->y，2是y->x->z，3是y->z->x，4是z->x->y，5是z->y->x
def convertCoordinate(x, y, z, sequence):
    mx = matrix([
        [1, 0, 0],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x)]
    ])
    my = matrix([
        [cos(y), 0, sin(y)],
        [0, 1, 0],
        [-sin(y), 0, cos(y)]
    ])
    mz = matrix([
        [cos(z), -sin(z), 0],
        [sin(z), cos(z), 0],
        [0, 0, 1]
    ])
    if sequence == 0:
        A = (mz.dot(my)).dot(mx)
    elif sequence == 1:
        A = (my.dot(mz)).dot(mx)
    elif sequence == 2:
        A = (mz.dot(mx)).dot(my)
    elif sequence == 3:
        A = (mx.dot(mz)).dot(my)
    elif sequence == 4:
        A = (my.dot(mx)).dot(mz)
    elif sequence == 5:
        A = (mx.dot(my)).dot(mz)
    return A



A1 = convertCoordinate(radians(0), radians(90), radians(-45), 5)
B1 = convertCoordinate(radians(45), -asin(1.0/sqrt(3))-pi/2.0, radians(0), 2)
A2 = convertCoordinate(radians(0), radians(0), radians(45), 2)
B2 = convertCoordinate(radians(-45), asin(1.0/sqrt(3))-pi, radians(0), 2)    
A3 = convertCoordinate(radians(180), radians(90), radians(-45), 1)
B3 = convertCoordinate(radians(135), asin(1.0/sqrt(3))-pi/2.0, radians(0), 2)
A4 = convertCoordinate(radians(45), radians(-90), radians(0), 3)
B4 = convertCoordinate(radians(45), asin(1.0/sqrt(3))-pi/2.0, radians(0), 2)

# get x,y,z coords from out image pixels coords
# i,j 是输出图像的xy坐标
# face 是面的号码
# halfOutSize 输出图像宽度的一半
# 将输出图像的xy值转换为三维空间中的xyz坐标，对应成一个坐标值由-1到1的立方体
def outImgToXYZ(i,j,face,halfOutSize,toward):
    if (toward == 0):  # front face
        a = i * 4.0 / (halfOutSize * 2)
        b = j * 4.0 / (halfOutSize * 2)
        if face==0: # down
            a = a - 2.0
            b = 2.0 - b
            (x,y,z) = (sqrt(3) - 1, a, b)
        elif face==1: # left top
            # first coordinate conversion
            c = A1.dot(matrix([b-1, a-1, 0]).T)
            c[2] = c[2] * sqrt(3)  # stretch
            # second coordinate conversion
            c[0] = c[0] - (1 - sqrt(3))
            c[1] = c[1] - 0
            c[2] = c[2] - sqrt(2)
            d = B1.dot(c)
            (x,y,z) = (d[0], d[1], d[2])
        elif face==2: # left bottom
            # first coordinate conversion
            c = A2.dot(matrix([b-3, a-1, 0]).T)
            c[0] = c[0] * sqrt(3)  # stretch
            # second coordinate conversion
            c[0] = c[0] - sqrt(2)
            c[1] = c[1] - 0
            c[2] = c[2] - (1 - sqrt(3))
            d = B2.dot(c)
            (x,y,z) = (d[0], d[1], d[2])
        elif face==3: # right top
            # first coordinate conversion
            c = A3.dot(matrix([b-1, a-3, 0]).T)
            c[2] = c[2] * sqrt(3)  # stretch
            # second coordinate conversion
            c[0] = c[0] - (sqrt(3) - 1)
            c[1] = c[1] - 0
            c[2] = c[2] - sqrt(2)
            d = B3.dot(c)
            (x,y,z) = (d[0], d[1], d[2])
        elif face==4: # right bottom
            # first coordinate conversion
            c = A4.dot(matrix([b-3, a-3, 0]).T)
            c[2] = c[2] * sqrt(3)  # stretch
            # second coordinate conversion
            c[0] = c[0] - (sqrt(3) - 1)
            c[1] = c[1] - 0
            c[2] = c[2] - sqrt(2)
            d = B4.dot(c)
            (x,y,z) = (d[0], d[1], d[2])
        return (x,y,z)

# convert using an inverse transformation
def convertBack(imgIn,imgOut):
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0]/4   # 视角宽度
    halfOutSize = outSize[0] / 2;
    for i in xrange(outSize[0]):
        for j in xrange(outSize[1]):
            face = 0
            if fabs(halfOutSize - i) + fabs(halfOutSize - j) <= halfOutSize:
                face = 0
            elif (i < halfOutSize) and (j < halfOutSize):
                face = 1    # 左上角
            elif (i < halfOutSize) and (j > halfOutSize):
                face = 2    # 左下角
            elif (i > halfOutSize) and (j < halfOutSize):
                face = 3    # 右上角
            elif (i > halfOutSize) and (j > halfOutSize):
                face = 4    # 右下角
            
            (x,y,z) = outImgToXYZ(i,j,face,halfOutSize, 0)
            theta = atan2(y,x) # 水平方向夹角
            r = hypot(x,y)
            phi = atan2(z,r) # 垂直方向夹角
            # 对应原图像的坐标值
            uf = ( 2.0*edge*(theta + pi) / pi )
            vf = ( 2.0*edge * (pi/2 - phi)/pi)
            outPix[i,j] = inPix[int(uf) % inSize[0], clip(vf,0,inSize[1]-1)]

imgIn = Image.open("2.jpg")
inSize = imgIn.size
imgOut = Image.new("RGB",((int)(inSize[0] / 4 * sqrt(2)) , (int)(inSize[0] / 4 * sqrt(2))),"black")
convertBack(imgIn,imgOut)
imgOut.save("out.jpg")
imgOut.show()