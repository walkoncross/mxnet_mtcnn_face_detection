# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 06:32:06 2017

@author: zhaoy
"""

import cv2


def cv2_put_text_to_image(img, text, x, y, font_pix_h=10, color=(255, 0, 0)):
    if font_pix_h < 10:
        font_pix_h = 10

    # print img.shape

    h = img.shape[0]

    if x < 0:
        x = 0

    if y > h - 1:
        y = h - font_pix_h

    if y < 0:
        y = font_pix_h

    font_size = font_pix_h / 30.0
    # print font_size
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, color, 1)