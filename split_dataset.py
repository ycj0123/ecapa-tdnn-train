#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:09:44 2020

@author: krishna, Iuthing
"""

import os
import random
import glob
import argparse
from tqdm import tqdm

def random_split(mylist, ratio):
    if ratio < 0.5:
        ratio = 1-ratio
    idx = int(ratio*len(mylist))
    random.shuffle(mylist)
    long = mylist[:idx]
    short = mylist[idx:]
    return long, short

# make a list of train, val and test data
def extract_split_files(folder_path, valid_frac, test_frac):
    all_lang_folders = sorted(glob.glob(folder_path+'/*/'))
    train_list = []
    val_list = []
    test_list = []

    for id, lang_folderpath in enumerate(all_lang_folders):
        all_files = sorted(glob.glob(lang_folderpath+'/*.wav'))
        lang_train, other = random_split(all_files, valid_frac+test_frac)
        larger = max(valid_frac, test_frac)
        lang_val, lang_test = random_split(other, larger/(valid_frac+test_frac))
        train_list += lang_train
        val_list += lang_val
        test_list += lang_test

    return train_list, test_list, val_list

# copy the data from origin
def make_split(file_list, split):
    for f in tqdm(file_list):
        parent = "/".join(f.split("/")[:-3])
        lang = f.split("/")[-2]
        fname = f.split("/")[-1]
        dir = os.path.join(parent, split, lang)
        os.makedirs(dir, exist_ok=True)
        cmd = f"cp {f} {dir}/{fname}"
        os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument('-d', "--data_path", default="/mnt/storage2t/vx_style_crnn/raw", type=str,help='Dataset path')
    parser.add_argument('-v', "--valid_frac", default="0.1", type=float, help="portion to split into valid set")
    parser.add_argument('-t', "--test_frac", default="0.1", type=float, help="portion to split into test set")
    config = parser.parse_args()
    train_list, test_list, val_list = extract_split_files(config.data_path, config.valid_frac, config.test_frac)
    
    make_split(train_list, "train")
    make_split(val_list, "val")
    make_split(test_list, "test")