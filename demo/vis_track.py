import argparse
import cv2
import os
import numpy as np
import motmetrics as mm
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SLADet.config import get_cfg
from detectron2.utils.logger import setup_logger

from tracker.multitracker import SLATracker

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./data/MOT/MOT16/test/',
        help='MOT test image sequence path',
    )

    parser.add_argument(
        '--sequence',
        type=str,
        default='MOT16-01',
        help='Specific sequence for testing',
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='output',
        help='Output path for saving result frames and .txt files',
    )

    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--min_box_area', type=float, default=200)

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

if __name__ == '__main__':
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # img data
    imgdata = os.listdir(args.dataset_path + args.sequence + '/img1')
    imgdata.sort()

    # # output img data
    # frame_dir = args.output_path + '/' + args.sequence + '/frame'
    # if not os.path.exists(frame_dir):
    #     os.makedirs(frame_dir)

    mot_tracker = SLATracker(cfg, args, 30)

    cv2.namedWindow('ss', cv2.WINDOW_NORMAL)
    total_time = 0

    with open(args.output_path + '/' + args.sequence + '/%s.txt' % (args.sequence), 'w') as out_file:
        frame = 0
        for filename in imgdata:
            frame += 1
            image = cv2.imread(args.dataset_path + args.sequence + '/img1/' + filename)

            bboxes, ids = [], []
            t1 = time.time()
            online_targets = mot_tracker.update(image)
            t2 = time.time()
            total_time += (t2 - t1)
            for t in online_targets:
                tlwh, tid = t.tlwh, t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    tlwh[2:] += tlwh[:2]
                    bboxes.append(tlwh)
                    ids.append(tid)

            for box, id in zip(bboxes, ids):
                color = compute_color_for_labels(id)
                label = '{}{:d}'.format("", id)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
                cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[0]) + t_size[0] + 3, int(box[1]) + t_size[1] + 4), color, -1)
                cv2.putText(image, label, (int(box[0]), int(box[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [255, 255, 255], 2)
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                frame, id, box[0], box[1], box[2] - box[0], box[3] - box[1]), file=out_file)  # 输出到文件

            cv2.imshow('ss', image)
            cv2.waitKey(1)

    print('Total Time: {}'.format(total_time))
    print('Done!')