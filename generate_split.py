import json
import numpy as np
import os

np.random.seed(2020) # to ensure you always get the same train/test split

data_path = '../data/RedLights2011_Medium'
gts_path = '../data/hw02_annotations'
split_path = '../data/hw02_splits'
os.makedirs(split_path, exist_ok=True) # create directory if needed

split_test = True # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []

'''
Your code below. 
'''
permutated_file_indices = np.random.permutation(len(file_names))
train_file_indices, test_file_indices = permutated_file_indices[: int(0.85*len(file_names))], \
                                        permutated_file_indices[int(0.85*len(file_names)):]
file_names_train, file_names_test = np.array(file_names)[train_file_indices], np.array(file_names)[test_file_indices]


assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'formatted_annotations_mturk.json'),'r') as f:
        gts = json.load(f)
    
    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}
    '''
    Your code below. 
    '''
    gts_l = list(gts.items())
    np.random.seed(2020)
    permutated_gts_indices = np.random.permutation(len(gts_l))
    train_gts_indices, test_gts_indices = permutated_gts_indices[: int(0.85 * len(gts_l))], permutated_gts_indices[int(0.85 * len(gts_l)):]
    gts_train, gts_test = np.array(gts_l)[train_gts_indices, :], np.array(gts_l)[test_gts_indices, :]
    gts_train = {gt[0]: gt[1] for gt in gts_train}
    gts_test = {gt[0]: gt[1] for gt in gts_test}

    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)
    
    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)