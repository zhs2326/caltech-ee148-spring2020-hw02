import os
import json
import matplotlib.pyplot as plt
import numpy as np

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    x1, y1, x2, y2 = box_1[1], box_1[0], box_1[3], box_1[2]
    x3, y3, x4, y4 = box_2[1], box_2[0], box_2[3], box_2[2]
    x5, y5, x6, y6 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)

    if x5 <= x6 and y5 <= y6:
        intersection = (x6 - x5) * (y6 - y5)
    else:
        intersection = 0

    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection

    iou = intersection / union
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.8):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        hit = set()

        for j in range(len(pred)):
            for i in range(len(gt)):
                iou = compute_iou(pred[j][:4], gt[i])
                if pred[j][4] >= conf_thr and iou >= iou_thr:
                    TP += 1
                    hit.add(i)
                    break
            else:
                if pred[j][4] >= conf_thr:
                    FP += 1

        FN += len(gt)-len(hit)


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train_multi_09_confthre_05_iou.json'),'r') as f:
    preds_train = json.load(f)

    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test_multi_09conf_05_iou.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

confidence_thrs = np.arange(0.99, 0.9, -0.002) # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
iou_thrs = [0.25,  0.5, 0.75]

for iou_thr in iou_thrs:
    precision = []
    recall = []
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)

    # Plot training set PR curves
        #print('tp', tp_train[i], conf_thr)
        #print('fp', fp_train[i], conf_thr)
        #print('fn', fn_train[i], conf_thr)
        if not tp_train[i]+fp_train[i]:
            precision.append(1)
        else:
            precision.append(tp_train[i]/(tp_train[i]+fp_train[i]))
        recall.append(tp_train[i]/(tp_train[i]+fn_train[i]))
        #print('precision', precision, conf_thr)
        #print('recall', recall, conf_thr)

    area = 0
    for i in range(1, len(recall)):
        area += (recall[i] - recall[i - 1]) * (precision[i - 1] + precision[i]) / 2
    print('area under PR curve on training set', area)

    if sum(recall):
        plt.plot(recall, precision, label='IOU='+str(iou_thr))
    else:
        plt.scatter(0, 0, marker='o', label='IOU='+str(iou_thr))
    plt.legend()

plt.title('weakened version PR curve on training set')
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()



if done_tweaking:
    print('Code for plotting test set PR curves.')

    confidence_thrs = np.arange(0.99, 0.9, -0.002)  # using (ascending) list of confidence scores as thresholds
    tp_test = np.zeros(len(confidence_thrs))
    fp_test = np.zeros(len(confidence_thrs))
    fn_test = np.zeros(len(confidence_thrs))
    iou_thrs = [0.25, 0.5, 0.75]

    for iou_thr in iou_thrs:
        precision = []
        recall = []
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thr,
                                                                   conf_thr=conf_thr)

            # Plot training set PR curves
            #print('tp', tp_test[i], conf_thr)
            #print('fp', fp_test[i], conf_thr)
            #print('fn', fn_test[i], conf_thr)
            if not tp_test[i] + fp_test[i]:
                precision.append(1)
            else:
                precision.append(tp_test[i] / (tp_test[i] + fp_test[i]))
            recall.append(tp_test[i] / (tp_test[i] + fn_test[i]))
            #print('precision', precision, conf_thr)
            #print('recall', recall, conf_thr)

        area = 0
        for i in range(1, len(recall)):
            area += (recall[i] - recall[i - 1]) * (precision[i - 1] + precision[i]) / 2
        print('area under PR curve on test set', area)

        if sum(recall):
            plt.plot(recall, precision, label='IOU=' + str(iou_thr))
        else:
            if iou_thr == 0.5:
                plt.scatter(0, 0, marker='o', label='IOU=' + str(iou_thr))
            else:
                plt.scatter(0, 0, marker='x', label='IOU=' + str(iou_thr))
        plt.legend()

    plt.title('weakened version PR curve on test set')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

