#!/usr/bin/env python
# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import sys

import os.path as osp
import json

from cv2_helper import cv2_put_text_to_image


def print_usage():
    print(
        '''
    python demo_test_list.py img_list_file save_dir save_image=<0, 1> show_image=<0, 1>
    '''
    )


def demo(list_fn, save_dir, save_img=True, show_img=False):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, 'mtcnn_fd_rlt.json'), 'w')
    result_list = []

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
        fp.close()
    else:
        file_list.append('test2.jpg')

    img_cnt = 0
    time_ttl = 0.0

    # process image file list
    for fn in file_list:
        fn = fn.strip()
        print('\n===> Processing image: ' + str(fn))

        if fn == '':
            print 'empty line, not a file name, skip to next'
            continue
        if fn[0] == '#':
            print 'skip line starts with #, skip to next'
            continue

        rlt = {}
        rlt["filename"] = fn
        rlt["faces"] = []
        rlt['face_count'] = 0

        print rlt

        img = cv2.imread(fn)
        if img is None:
            print('---> Failed to load image, skip to next one')
            rlt["message"] = "failed to load"
            result_list.append(rlt)
            continue

        t1 = time.clock()

        # run detector
        results = detector.detect_face(img)
        t2 = time.clock()
        print("---> detect_face() cost %f seconds" % (t2 - t1))

        img_cnt += 1
        time_ttl += t2 - t1

        print("\n===> Processing %d images cost %f seconds, avg time: %f seconds/image" %
              (img_cnt, time_ttl, time_ttl / img_cnt))

        if results is not None:

            bboxes = results[0]
            points = results[1]

            for (box, pts) in zip(bboxes, points):
                box = box.tolist()
                pts = pts.tolist()
                tmp = {'rect': box[0:4],
                       'score': box[4],
                       'pts': pts
                       }
                rlt['faces'].append(tmp)

            rlt['face_count'] = len(bboxes)

            # extract aligned face chips
        #    chips = detector.extract_image_chips(img, points, 144, 0.37)
        #    for i, chip in enumerate(chips):
        #        #cv2.imshow('chip_'+str(i), chip)
        #        cv2.imwrite('chip_'+str(i)+'.png', chip)
            if save_img or show_img:
                draw = img.copy()
                for b in bboxes:
                    cv2.rectangle(draw, (int(b[0]), int(b[1])),
                                  (int(b[2]), int(b[3])), (255, 255, 255))

                    text = '%2.3f' % (b[4]*100)
                    cv2_put_text_to_image(draw, text, int(b[0]), int(b[3]) + 5 )

                for p in points:
                    for i in range(5):
                        cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), -1)

            if save_img:
                base_name = osp.basename(fn)
                save_name = osp.join(save_dir, base_name)
                cv2.imwrite(save_name, draw)

        rlt['message'] = 'success'
        result_list.append(rlt)

        if show_img:
            cv2.imshow("detection result", draw)
            key = cv2.waitKey(0)
            # print key
            if key == 27:
                break

    json.dump(result_list, fp_rlt, indent=4)
    fp_rlt.close()

    if show_img:
        cv2.destroyAllWindows()


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

if __name__ == '__main__':
    print_usage()

    list_fn = './list_img.txt'
    save_dir = './test_save_dir'
#    list_fn = './list_lfw_failed3.txt'
#    save_dir = './test_save_dir_lfw'

    if len(sys.argv) > 1:
        list_fn = sys.argv[1]

    if len(sys.argv) > 2:
        save_dir = sys.argv[2]

    demo(list_fn, save_dir)
