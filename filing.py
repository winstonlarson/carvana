import os, shutil
from sklearn.model_selection import train_test_split
import pandas as pd

dataset_dir = '/home/ubuntu/carvana/input/data/'
image_dir = dataset_dir + 'images/train'
mask_dir = dataset_dir + 'masks/train_masks'

new_base_dir = 'home/ubuntu/carvana/input/'
train_dir = new_base_dir + 'train'
test_dir = new_base_dir + 'test'

train_image_dir = train_dir + 'images/train_images'
train_mask_dir = train_dir + 'masks/train_masks'
test_image_dir = test_dir + 'images/test_images'
test_mask_dir = test_dir + 'masks/test_masks'

df_train = pd.read_csv('input/data/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])
ids_train_split, ids_test_split = train_test_split(ids_train, test_size=0.2, random_state=42)

i = 0
for train_id in ids_train_split:
    image_filename = train_id + '.jpg'
    mask_filename = train_id + '.png'
    image_src = os.path.join(image_dir, image_filename)
    image_dst = os.path.join(train_image_dir, image_filename)
    mask_src = os.path.join(mask_dir, mask_filename)
    mask_dst = os.path.join(train_mask_dir, mask_filename)
    shutil.copyfile(image_src, image_dst)
    shutil.copyfile(mask_src, mask_dst)
    i += 1
    if i % 10 == 0:
        print('copied 10')

i = 0
for test_id in ids_test_split:
    image_filename = test_id + '.jpg'
    mask_filename = test_id + '.png'
    image_src = os.path.join(image_dir, image_filename)
    image_dst = os.path.join(test_image_dir, image_filename)
    mask_src = os.path.join(mask_dir, mask_filename)
    mask_dst = os.path.join(test_mask_dir, mask_filename)
    shutil.copyfile(image_src, image_dst)
    shutil.copyfile(mask_src, mask_dst)
    i += 1
    if i % 10 == 0:
        print('copied 10')