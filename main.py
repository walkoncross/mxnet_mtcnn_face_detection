# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

import os.path as osp

list_fn = './list_img.txt'
save_dir = './test_save_dir'

if not osp.exists(save_dir):
    os.makedirs(save_dir)

t1 = time.clock()

# create a detector
detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(
    0), num_worker=1, accurate_landmark=False)

t2 = time.clock()
print("\n===> Init detector cost %f seconds" % (t2 - t1))

file_list = []

if osp.exists(list_fn):
    with open(list_fn, 'r') as fp:
        for line in fp:
            line = line.strip()
            if len(line) > 0:
                file_list.append(line)
else:
    file_list.append('test2.jpg')

img_cnt = 0
time_ttl = 0.0

# process image file list
for fn in file_list:
    print('\n===> Processing image: ' + fn)
    img = cv2.imread(fn)
    if img is None:
        print('---> Failed to load image, skip to next one')
        continue

    t1 = time.clock()

    # run detector
    results = detector.detect_face(img)
    t2 = time.clock()
    print("---> detect_face() cost %f seconds" % (t2 - t1))

    img_cnt += 1
    time_ttl += t2 - t1

    if results is not None:

        total_boxes = results[0]
        points = results[1]

        # extract aligned face chips
    #    chips = detector.extract_image_chips(img, points, 144, 0.37)
    #    for i, chip in enumerate(chips):
    #        #cv2.imshow('chip_'+str(i), chip)
    #        cv2.imwrite('chip_'+str(i)+'.png', chip)

        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])),
                          (int(b[2]), int(b[3])), (255, 255, 255))

        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

        base_name = osp.basename(fn)
        save_name = osp.join(save_dir, base_name)
        cv2.imwrite(save_name, draw)
        cv2.imshow("detection result", draw)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
      #print key
        if key==27:
            break

    print("\n===> Processing %d images cost %f seconds, avg time: %f seconds/image" %
          (img_cnt, time_ttl, time_ttl / img_cnt))


# --------------
# test on camera
# --------------
'''
camera = cv2.VideoCapture(0)
while True:
    grab, frame = camera.read()
    img = cv2.resize(frame, (320,180))

    t1 = time.time()
    results = detector.detect_face(img)
    print 'time: ',time.time() - t1

    if results is None:
        continue

    total_boxes = results[0]
    points = results[1]

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
    cv2.imshow("detection result", draw)
    cv2.waitKey(30)
'''
