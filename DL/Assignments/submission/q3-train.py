import argparse
import collections

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
from matplotlib import pyplot as plt
import csv, json
import os
import pandas as pd
from numpy.random import RandomState

data_dir = "/content/drive/My Drive/Deep_Learning_Assignments/Assignment1/Q3/assignment-data/data/"

annotations_json_path = data_dir + "train_annotations.json"
annotations_csv_path = data_dir + "train_annotations_csv.csv"

val_csv = annotations_csv_path
train_csv = data_dir + "train.csv"


weights_path = data_dir + "pretrained_weights.pt"
images_path = data_dir + "train"
# os.system("unzip -q \"{}\" -d \"{}\"".format(images_path,data_dir))

best_model_path = "/content/drive/My Drive/Deep_Learning_Assignments/Assignment1/Q3/dlassignment1/q3/freeze_model/third/csv_retinanet_best_model_mAP.pt"
class_list = data_dir + "class_list.csv"

# Clean the json
with open(annotations_json_path) as f:
  annot_json = json.load(f)

annot_json_cleaned = {}

annot_json_cleaned["info"] = annot_json["info"]
annot_json_cleaned["categories"] = annot_json["categories"]
annot_json_cleaned["images"] = []
annot_json_cleaned["annotations"] = []

file_path = {}
for path, subdirs, files in os.walk(images_path):
    if files:
        # print(os.path.join(path, min(files)))
        for file in files:
          # print(file)
          file_path[file] = os.path.join(path, file)

def inFolder(elem):
  if elem in file_path:
    return True
  return False

# remove images which arent there
for elem in annot_json["images"]:
  if inFolder(elem["file_name"]):
    annot_json_cleaned["images"].append(elem)

for elem in annot_json["annotations"]:
  if inFolder(elem["image_id"]+".jpg"):
    annot_json_cleaned["annotations"].append(elem)

annot_json = annot_json_cleaned

print("JSON cleaned")

# Generate class list
f_csv = open(class_list, 'w', newline='')

writer = csv.writer(f_csv)
lines = [
["bird","0"],
["bobcat","1"],
["car","2"],
["cat","3"],
["raccoon","4"],
["rabbit","5"],
["coyote","6"],
["squirrel","7"]
]
for line in lines:
  writer.writerow(line)

print("Made class list")
# Generate annotation CSV
class_map = {
11:"bird",
6:"bobcat",
33:"car",
16:"cat",
3:"raccoon",
10:"rabbit",
9:"coyote",
5:"squirrel"
}

f_csv = open(annotations_csv_path, 'w', newline='')
writer = csv.writer(f_csv)

for anno in annot_json["annotations"]:
  bbox = [int(x) for x in anno["bbox"]]
  bbox[2] = bbox[0] + bbox[2]
  bbox[3] = bbox[1] + bbox[3]
  if(anno["category_id"] in class_map):
    class_name = class_map[anno["category_id"]]
    img_path = images_path +"/"+class_name+"/" +anno["image_id"] + ".jpg"
    # print(img_path)
    writer.writerow([img_path,bbox[0],bbox[1],bbox[2],bbox[3],class_name])

print("Made annot CSV")

# Load the best model
retinanet = torch.load(best_model_path)
print("Loaded best model")
# Load val dataset
dataset_val = CSVDataset(train_file = val_csv, class_list=class_list,
                                    transform=transforms.Compose([Normalizer(),Resizer()]))
print("Loaded val dataset")
# Eval karo
use_gpu = True

if use_gpu:
    retinanet = retinanet.cuda()
retinanet = torch.nn.DataParallel(retinanet).cuda()

retinanet.training = False
retinanet.eval()
retinanet.module.freeze_bn()

mAP = csv_eval.evaluate(dataset_val, retinanet)

def avg(mAP):
  avg_map = 0
  for key in mAP:
    avg_map += mAP[key][0]
  avg_map /= 8

  print("Avg mAP",avg_map)

  return avg_map
print(avg(mAP))
print(mAP)