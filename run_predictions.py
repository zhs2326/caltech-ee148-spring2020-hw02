import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image


def normalize(I):
    I = np.copy(I)
    mean = np.mean(I)
    std = np.std(I)

    if std:
        I = (I-mean)/std
    else:
        I = np.ones(I.shape)

    return I


def non_maximum_suppression(candidates, IOU_threshold=0.5):
    if candidates is None:
        return None
    candidates = sorted(candidates, key=lambda x: x[-1], reverse=True)
    outputs = []

    for candidate in candidates:
        x1, y1, x2, y2, score = candidate[1], candidate[0], candidate[3], candidate[2], candidate[4]
        for output in outputs:
            x3, y3, x4, y4 = output[1], output[0], output[3], output[2]
            #calculate the IOU here
            x5, y5, x6, y6 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
            if x5 <= x6 and y5 <= y6:
                intersection = (x6 - x5) * (y6 - y5)
            else:
                intersection = 0
            union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
            IOU = intersection / union

            if IOU >= IOU_threshold:
                break
        else:
            outputs.append([y1, x1, y2, x2, score])

    return outputs


def compute_convolution(I, T, padding=0, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    template_rows, template_cols, template_channels = np.shape(T)

    heatmap = np.random.random([(n_rows-template_rows+2*padding)//stride+1, (n_cols-template_cols+2*padding)//stride+1])

    I_with_padding = np.zeros([n_rows+2*padding, n_cols+2*padding, n_channels])
    I_with_padding[padding: padding+n_rows, padding: padding+n_cols] = I

    for i in range(0, I_with_padding.shape[0]-template_rows+1, stride):
        for j in range(0, I_with_padding.shape[1]-template_cols+1, stride):
            window = I_with_padding[i: i+template_rows, j: j+template_cols]
            window_normalized = normalize(window)
            heatmap[i//stride, j//stride] = np.sum(np.multiply(T, window_normalized))/(template_rows*template_cols*template_channels)
            heatmap[i//stride, j//stride] = (1+heatmap[i//stride, j//stride])/2



    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap, I, T, padding=0, stride=1, conf_thr=0.8):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            tl_row = max(i*stride-padding, 0)
            tl_col = max(j*stride-padding, 0)
            br_row = min(i*stride-padding+T.shape[0], I.shape[0]-1)
            br_col = min(j*stride-padding+T.shape[1], I.shape[1]-1)
            score = heatmap[i, j]
            if score >= conf_thr:
                output.append([tl_row, tl_col, br_row, br_col, score])

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # You may use multiple stages and combine the results
    T = np.array(Image.open('../data/RedLights2011_Medium/RL-001.jpg'))
    T = T[154: 161, 316: 322]
    T = normalize(T)

    heatmap = compute_convolution(I, T, padding=0, stride=1)
    output = predict_boxes(heatmap, I, T, padding=0, stride=1, conf_thr=0.9)

    IOU_threshold = 0.5
    output = non_maximum_suppression(output, IOU_threshold)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''

preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)
    print(i)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        print(i)

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)


        preds_test[file_names_test[i]] = detect_red_light_mf(I)


    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
