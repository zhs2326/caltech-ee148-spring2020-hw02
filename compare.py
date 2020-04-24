import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image

preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

with open(os.path.join(preds_path, 'preds_test_multi_09conf.json'), 'r') as f:
    preds_test = json.load(f)

with open(os.path.join(gts_path, 'annotations_test.json'), 'r') as f:
    gts_test = json.load(f)


img_name = 'RL-149.jpg'
img_path = '../data/RedLights2011_Medium/'+img_name
img = Image.open(img_path)
img = np.array(img)

pred_bounding_boxes = preds_test[img_name]
truth_bounding_boxes = gts_test[img_name]

fig, ax = plt.subplots(1)
ax.imshow(img)

for bounding_box in pred_bounding_boxes:
    x1, y1, x2, y2 = bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='yellow',
                             facecolor='none')
    ax.add_patch(rect)

for bounding_box in truth_bounding_boxes:
    x1, y1, x2, y2 = bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red',
                             facecolor='none')
    ax.add_patch(rect)
plt.show()

