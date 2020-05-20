# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
from fvcore.common.file_io import PathManager
import subprocess
from collections import OrderedDict

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .evaluator import DatasetEvaluator

DEFAULT_TRUNCATED = 0.0 # 0% truncated
DEFAULT_OCCLUDED = 0    # fully visible

CLASS_NAMES = [
    'car', 'van', 'truck', 'pedestrian', 'pedestrian',
    'cyclist', 'tram', 'misc', 'dontcare', 'person_sitting'
]

class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            print(classes)
            print(scores)
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            print(current_file_dir)
            path = os.path.join(current_file_dir, 'eval_kitti/build/results/exp1/data/')
            os.makedirs(path, exist_ok=True)
            with open(path + image_id + ".txt", "w") as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
                for box, score, cls in zip(boxes, scores, classes):
                    xmin, ymin, xmax, ymax = box
                    # The inverse of data loading logic in `datasets/pascal_voc.py`
                    xmin += 1
                    ymin += 1
                    detection = {}
                    detection['label'] = CLASS_NAMES[cls]
                    kitti_row = [-1] * 16
                    kitti_row[0] = detection['label']
                    kitti_row[1] = DEFAULT_TRUNCATED
                    kitti_row[2] = DEFAULT_OCCLUDED
                    kitti_row[4:8] = xmin, ymin, xmax, ymax
                    kitti_row[15] = score
                    csvwriter.writerow(kitti_row)

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        bashCommand = "bash ./run_evaluation.sh"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        output_lines = output.decode("utf-8")
        split_word = ' (%): '
        metrics = OrderedDict()
        for line in output_lines.split('\n'):
            if split_word in line:
                metrics_name, val = line.split(split_word)
                easy, medium, hard = val.split(' / ')
                metrics[metrics_name] = {'easy': float(easy), 'medium': float(medium), 'hard': float(hard)}
        return metrics
