# Author: Martin Sundermeyer (martin.sundermeyer@dlr.de)
# Robotics Institute at DLR, Department of Perception and Cognition

"""Calculates Instance Mask Annotations in Coco Format."""

import numpy as np
import os
import datetime
import json

from bop_toolkit_lib import pycoco_utils
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'jmas',

  # Dataset split. Options: 'train', 'test'.
  'dataset_split': 'train',

  # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # bbox type. Options: 'modal', 'amodal'.
  'bbox_type': 'amodal',

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

}
################################################################################

datasets_path = p['datasets_path']
dataset_name = p['dataset']
split = p['dataset_split']
split_type = p['dataset_split_type']
bbox_type = p['bbox_type']

dp_split = dataset_params.get_split_params(datasets_path, dataset_name, split, split_type=split_type)
dp_model = dataset_params.get_model_params(datasets_path, dataset_name)

complete_split = split
if dp_split['split_type'] is not None:
    complete_split += '_' + dp_split['split_type']

CATEGORIES = [{'id': obj_id, 'name':str(obj_id), 'supercategory': dataset_name} for obj_id in dp_model['obj_ids']]
INFO = {
    "description": dataset_name + '_' + split,
    "url": "https://github.com/thodan/bop_toolkit",
    "version": "0.1.0",
    "year": datetime.date.today().year,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

# initial empty COCO dict, will serve as running data to iteratively merge into
merged_coco_dict = {
  "info": INFO,
  "licenses": [],
  "categories": CATEGORIES,
  "images": [],
  "annotations": []
}

for scene_id in dp_split['scene_ids']:

  # coco file for this scene
  coco_gt_path = dp_split['scene_gt_coco_tpath'].format(scene_id=scene_id)
  if bbox_type == 'modal':
    coco_gt_path = coco_gt_path.replace('scene_gt_coco', 'scene_gt_coco_modal')

  # load coco annotation dictionary
  try:
    with open(coco_gt_path, 'r') as fp:
      coco_scene_dict = json.load(fp)
  except FileNotFoundError:
    misc.log(f"Scene ID {scene_id} missing")
    continue

  # change file paths to use new image root
  for i in range(len(coco_scene_dict['images'])):
    fname = coco_scene_dict['images'][i]['file_name']
    new_fname = os.path.join(p['dataset_split'], f'{scene_id:06d}', fname)
    coco_scene_dict['images'][i]['file_name'] = new_fname

  misc.log('Merging Coco Annotations - dataset: {} ({}, {}), scene: {}'.format(
        p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id))

  # merge with running coco dict
  merged_coco_dict, _ = pycoco_utils.merge_coco_annotations(merged_coco_dict, coco_scene_dict)


# write out final COCO file
merged_coco_file = os.path.join(dp_split['base_path'], f"coco_{dp_split['split']}.json")
misc.log('Saving Merged Coco File to {}'.format(
         merged_coco_file))
with open(merged_coco_file, 'w') as fp:
  json.dump(merged_coco_dict, fp, indent=2)
