import os
import platform
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, trim_zeros
import numpy as np

# https://github.com/pytorch/pytorch/issues/3678
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path+'/yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import yaml
import solution
import main 

import pdb
from matplotlib import pyplot as plt

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def detect(opt, device, half, colorDict, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, skipLimit = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.skip_frames
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    groundtruths_path = opt.groundtruths
    colorOrder = [color for color in colorDict]
    frame_num = 0
    framestr = 'Frame {frame}'
    fpses = []
    frame_catch_pairs = []
    ball_person_pairs = {}
    id_mapping = {}

    for color in colorDict:
        ball_person_pairs[color] = 0
    
    print("FRAMES SKIPPED: " + str(skipLimit))

    # Read Class Name Yaml
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    names = data_dict['names']

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(dir_path + '/' + cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # Initialize
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    #Skip Variables
    skipThreshold = 0 #Current number of frames skipped

    # gt ball/person number
    gt_balls, gt_people  = solution.gt_balls_people_cnt(groundtruths_path)
    detected_balls, detected_people = 0, 0
    data_xyxys = np.array([])

    for path, img, im0s, vid_cap in dataset:
        if frame_num > 10 and skipThreshold < skipLimit:
            skipThreshold = skipThreshold + 1
            frame_num += 1
            continue
        
        skipThreshold = 0


        #pdb.set_trace()
        #cv2.imshow(path,im0s)

        # 잘 탐지한 경우 img 자르기
        if gt_balls==detected_balls and gt_people==detected_people:
            im0s_h, im0s_w = im0s.shape[0], im0s.shape[1]
            x_, y_, w_, h_ = for_im_trim(data_xyxys, im0s_w, im0s_h, path, im0s)
            #cv2.imshow(path, im_trim(im0s, , 0, im0s_w, im0s_h))
            """
            cv2.imshow(path, im_trim(im0s, x_, y_, w_, h_))
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
            """
            im0_ = im_trim(im0s, x_, y_, w_, h_)
            img = letterbox(im0_, imgsz, stride=stride)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

        """
        else:
            cv2.imshow(path, im0s)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        """
        
        
        # 잘 탐지하지 못한 경우 안 자르고 그대로

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                if gt_balls==detected_balls and gt_people==detected_people:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_.shape).round()
                else:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                bbox_xywh = []
                confs = []
                clses = []

                # Write results
                for *xyxy, conf, cls in det:
                    #pdb.set_trace()
                    # img를 잘랐던 경우 좌표에 값 더해주기
                    if gt_balls==detected_balls and gt_people==detected_people:
                        x_, y_, w_, h_ = for_im_trim(data_xyxys, im0s_w, im0s_h, path, im0)
                        
                        # pdb.set_trace()
                        
                        xyxy[0] += x_
                        xyxy[1] += y_
                        xyxy[2] += x_
                        xyxy[3] += y_
                        
                        """
                        x_m = min(int(xyxy[0].item()), int(xyxy[2].item()))
                        y_m = min(int(xyxy[1].item()), int(xyxy[3].item()))
                        x_M = max(int(xyxy[0].item()), int(xyxy[2].item()))
                        y_M = max(int(xyxy[1].item()), int(xyxy[3].item()))
                        w_t = abs(x_M - x_m)
                        h_t = abs(y_M - y_m)
                        
                        cv2.imshow(path, im_trim(im0, int(x_m), int(y_m), int(w_t), int(h_t)))
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration
                        """


                    img_h, img_w, _ = im0.shape  # get image shape
                    x_c, y_c, bbox_w, bbox_h = main.bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    clses.append([cls.item()])
                    
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                clses = torch.Tensor(clses)
                # Pass detections to deepsort
                outputs = []
                if not 'disable' in groundtruths_path:
                    # print('\nenabled', groundtruths_path)
                    groundtruths = solution.load_labels(groundtruths_path, img_w,img_h, frame_num)
                    if (groundtruths.shape[0]==0):
                        outputs = deepsort.update(xywhs, confss, clses, im0)
                    else:
                        # print(groundtruths)
                        xywhs = groundtruths[:,2:]
                        tensor = torch.tensor((), dtype=torch.int32)
                        confss = tensor.new_ones((groundtruths.shape[0], 1))
                        clses = groundtruths[:,0:1]
                        outputs = deepsort.update(xywhs, confss, clses, im0)
                    
                    
                    if frame_num >= 2:
                        for real_ID in groundtruths[:,1:].tolist():
                            for DS_ID in xyxy2xywh(outputs[:, :5]):
                                if (abs(DS_ID[0]-real_ID[1])/img_w < 0.005) and (abs(DS_ID[1]-real_ID[2])/img_h < 0.005) and (abs(DS_ID[2]-real_ID[3])/img_w < 0.005) and(abs(DS_ID[3]-real_ID[4])/img_w < 0.005):
                                    id_mapping[DS_ID[4]] = int(real_ID[0])
                else:
                    outputs = deepsort.update(xywhs, confss, clses, im0)
                
                # count detected balls/people
                detected_balls = 0
                detected_people = 0

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    clses = outputs[:, 5]
                    scores = outputs[:, 6]

                    for cls in clses:
                    # ball
                        if cls == 1:
                           detected_balls += 1
                        elif cls == 0:
                           detected_people += 1

                    data_xyxys = outputs[:, :4]

                    
                    #Temp solution to get correct id's 
                    mapped_id_list = []
                    for ids in identities:
                        if(ids in id_mapping):
                            mapped_id_list.append(int(id_mapping[ids]))
                        else:
                            mapped_id_list.append(ids)

                    ball_detect, frame_catch_pairs, ball_person_pairs = solution.detect_catches(im0, bbox_xyxy, clses, mapped_id_list, frame_num, colorDict, frame_catch_pairs, ball_person_pairs, colorOrder, save_img)

                    t3 = time_synchronized()
                    if (save_img):
                        main.draw_boxes(im0, bbox_xyxy, [names[i] for i in clses], scores, ball_detect, id_mapping, identities)
                else:
                    t3 = time_synchronized()


            #Inference Time
            fps = (1/(t3 - t1))
            fpses.append(fps)
            print('FPS=%.2f' % fps)
            
            
            # Stream results
            if view_img:
                #im0 = im_trim(im0, 700, 700, 500, 500)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    #Draw frame number
                    tmp = framestr.format(frame = frame_num)
                    t_size = cv2.getTextSize(tmp, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                    cv2.putText(im0, tmp, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
            frame_num += 1
                    
        
    avgFps = (sum(fpses) / len(fpses))
    print('Average FPS = %.2f' % avgFps)


    outpath = os.path.basename(source)
    outpath = outpath[:-4]
    outpath = out + '/' + outpath + '_out.csv'
    print(outpath)
    solution.write_catches(outpath, frame_catch_pairs, colorDict, colorOrder)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    return

def im_trim(img, x, y, w, h):
    imgtrim= img[y:y+h, x:x+w]
    return imgtrim


def for_im_trim(data_xyxys, img_w, img_h, path, im0s):
    x_m, y_m, x_M, y_M = img_w, img_h, 0, 0
    tmp_w, tmp_h = 0, 0

    for xyxy in data_xyxys:
        x1, y1, x2, y2 = [int(i) for i in xyxy]
        x_m = min(x_m, x1, x2)
        y_m = min(y_m, y1, y2)
        x_M = max(x_M, x1, x2)
        y_M = max(y_M, y1, y2)

        tmp_w = max(tmp_w, abs(x2-x1))
        tmp_h = max(tmp_h, abs(y2-y1))

    x1 = x_m - tmp_w//2 if x_m - tmp_w//2 > 0 else 0
    y1 = y_m - tmp_w//2 if y_m - tmp_w//2 > 0 else 0
    x2 = x_M + tmp_w//2 if x_M + tmp_w//2 < img_w else img_w
    y2 = y_M + tmp_w//2 if y_M + tmp_w//2 < img_h else img_h
    x_, y_ = x1, y1
    w_, h_ = abs(x2-x1), abs(y2-y1)

    return [x_, y_, w_, h_]