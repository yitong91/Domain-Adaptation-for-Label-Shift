import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io
from utils import ToOneHot
import os
import pdb
def LoadDatasets(data_dir = './', sets={'mnist':1, 'svhn':1, 'mnistm':1, 'synth':1}):
    datasets = {}
    for key in sets.keys():
        datasets[key] = {}
    if 'mnist' in sets:
        mat = scipy.io.loadmat(data_dir + 'mnist.mat')
        datasets['mnist']['train'] = {'images': mat['train'].astype('float32'), 'labels': mat['labtrain']}
        datasets['mnist']['valid'] = {'images': mat['val'].astype('float32'), 'labels': mat['labval']}
        datasets['mnist']['test'] = {'images': mat['test'].astype('float32'), 'labels': mat['labtest']}  
    
    if 'svhn' in sets:
        mat = scipy.io.loadmat(data_dir + 'svhn.mat')
        datasets['svhn']['train'] = {'images': mat['train'].astype('float32'), 'labels': mat['labtrain']}
        datasets['svhn']['valid'] = {'images': mat['val'].astype('float32'), 'labels': mat['labval']}
        datasets['svhn']['test'] = {'images': mat['test'].astype('float32'), 'labels': mat['labtest']}  
    

    if 'synth' in sets:
        mat = scipy.io.loadmat(data_dir + 'synth.mat')
        datasets['synth']['train'] = {'images': mat['train'].astype('float32'), 'labels': mat['labtrain']}
        datasets['synth']['valid'] = {'images': mat['val'].astype('float32'), 'labels': mat['labval']}
        datasets['synth']['test'] = {'images': mat['test'].astype('float32'), 'labels': mat['labtest']}  
    
    if 'mnistm' in sets:
        mat = scipy.io.loadmat(data_dir + 'mnistm.mat')
        datasets['mnistm']['train'] = {'images': mat['train'].astype('float32'), 'labels': mat['labtrain']}
        datasets['mnistm']['valid'] = {'images': mat['val'].astype('float32'), 'labels': mat['labval']}
        datasets['mnistm']['test'] = {'images': mat['test'].astype('float32'), 'labels': mat['labtest']}  
        
    return datasets

def SetsConcatenate(datasets, sets):
    N_train = 0
    N_valid = 0
    N_test = 0
    
    for key in sets:
        N_train = N_train + datasets[key]['train']['images'].shape[0]
        N_valid = N_valid + datasets[key]['valid']['images'].shape[0]
        N_test = N_test + datasets[key]['test']['images'].shape[0]
        S = datasets[key]['train']['images'].shape[1]
    train = {'images': np.zeros((N_train,S,S,3)).astype(np.float32),'labels':np.zeros((N_train,10)).astype('float32'),'domains':np.zeros((N_train,)).astype('float32')}
    valid = {'images': np.zeros((N_valid,S,S,3)).astype(np.float32),'labels':np.zeros((N_valid,10)).astype('float32'),'domains':np.zeros((N_valid,)).astype('float32')}
    test = {'images': np.zeros((N_test,S,S,3)).astype(np.float32),'labels':np.zeros((N_test,10)).astype('float32'),'domains':np.zeros((N_test,)).astype('float32')}
    srt = 0
    edn = 0
    for key in sets:
        domain = sets[key]
        srt = edn
        edn = srt + datasets[key]['train']['images'].shape[0]
        train['images'][srt:edn,:,:,:] = datasets[key]['train']['images']
        train['labels'][srt:edn,:] = datasets[key]['train']['labels']
        train['domains'][srt:edn] = domain * np.ones((datasets[key]['train']['images'].shape[0],)).astype('float32')
    srt = 0
    edn = 0
    for key in sets:
        domain = sets[key]
        srt = edn
        edn = srt + datasets[key]['valid']['images'].shape[0]
        valid['images'][srt:edn,:,:,:] = datasets[key]['valid']['images']
        valid['labels'][srt:edn,:] = datasets[key]['valid']['labels']
        valid['domains'][srt:edn] = domain * np.ones((datasets[key]['valid']['images'].shape[0],)).astype('float32')
    srt = 0
    edn = 0
    for key in sets:
        domain = sets[key]
        srt = edn
        edn = srt + datasets[key]['test']['images'].shape[0]
        test['images'][srt:edn,:,:,:] = datasets[key]['test']['images']
        test['labels'][srt:edn,:] = datasets[key]['test']['labels']
        test['domains'][srt:edn] = domain * np.ones((datasets[key]['test']['images'].shape[0],)).astype('float32')
    return train, valid, test

def SourceTarget(datasets, sources, targets, unify_source = False):
    N1 = len(sources.keys())
    N_domain = N1 + len(targets.keys())
    domain_idx = 0
    for key in sources.keys():
        sources[key] = domain_idx
        domain_idx = domain_idx + 1
    for key in targets.keys():
        targets[key] = domain_idx    
        domain_idx = domain_idx + 1
        
    source_train, source_valid, source_test = SetsConcatenate(datasets, sources)
    target_train, target_valid, target_test = SetsConcatenate(datasets, targets)
    

    if unify_source:
        source_train['domains'] = ToOneHot(0 * source_train['domains'], 2)
        source_valid['domains'] = ToOneHot(0 * source_valid['domains'], 2)
        source_test['domains'] = ToOneHot(0 * source_test['domains'], 2)
        target_train['domains'] = ToOneHot(0 * target_train['domains'] + 1, 2)
        target_valid['domains'] = ToOneHot(0 * target_valid['domains'] + 1, 2)
        target_test['domains'] = ToOneHot(0 * target_test['domains'] + 1, 2)

        for key in sources.keys() + targets.keys():
            for s in ['train', 'valid','test']:
                datasets[key][s]['domains'] = np.zeros([datasets[key]['train']['images'].shape[0], 2])
                if key in sources.keys():
                    datasets[key][s]['domains'][:,0] = 1.0
                else:
                    datasets[key][s]['domains'][:,1] = 1.0


    else:
        source_train['domains'] = ToOneHot(source_train['domains'], N_domain)
        source_valid['domains'] = ToOneHot(source_valid['domains'], N_domain)
        source_test['domains'] = ToOneHot(source_test['domains'], N_domain)
        target_train['domains'] = ToOneHot(target_train['domains'], N_domain)
        target_valid['domains'] = ToOneHot(target_valid['domains'], N_domain)
        target_test['domains'] = ToOneHot(target_test['domains'], N_domain)
        for key in sources.keys() + targets.keys():
            for s in ['train', 'valid','test']:
                datasets[key][s]['domains'] = np.zeros([datasets[key]['train']['images'].shape[0], N_domain])
                if key in sources.keys():
                    datasets[key][s]['domains'][:,sources[key]] = 1.0
                else:
                    datasets[key][s]['domains'][:,targets[key]] = 1.0
    return datasets, source_train, source_valid, source_test, target_train, target_valid, target_test

def Normalize(data):
    image_mean = np.expand_dims(np.expand_dims(data.mean((1,2)),1),1)
    data = data - image_mean
    image_std = np.sqrt((data**2).mean((1,2))+0.00001)
    return data / np.expand_dims(np.expand_dims(image_std,1),1)

def NormalizeDataset(datasets, t = 'norm'):
    if t is 'mean':
        temp_data = []
        for key in datasets.keys():
            temp_data.append(datasets[key]['train']['images'])
        temp_data = np.concatenate(temp_data)
        image_mean = temp_data.mean((0, 1, 2))
        image_mean = image_mean.astype('float32')
        for key in datasets.keys():
            datasets[key]['train']['images'] = (datasets[key]['train']['images'].astype('float32') - image_mean)/255.
            datasets[key]['valid']['images'] = (datasets[key]['valid']['images'].astype('float32')  - image_mean)/255.
            datasets[key]['test']['images'] = (datasets[key]['test']['images'].astype('float32')  - image_mean)/255.
    elif t is 'standard':
        for key in datasets.keys():
            datasets[key]['train']['images'] = (datasets[key]['train']['images'].astype('float32'))/255.
            datasets[key]['valid']['images'] = (datasets[key]['valid']['images'].astype('float32'))/255.
            datasets[key]['test']['images'] = (datasets[key]['test']['images'].astype('float32'))/255.
    elif t is 'none':
        datasets = datasets
    elif t is 'individual':
        for key in datasets.keys():
            temp_data = datasets[key]['train']['images']
            image_mean = temp_data.mean((0, 1, 2))
            image_mean = image_mean.astype('float32')
            datasets[key]['train']['images'] = (datasets[key]['train']['images'].astype('float32') - image_mean)/255.
            datasets[key]['valid']['images'] = (datasets[key]['valid']['images'].astype('float32')  - image_mean)/255.
            datasets[key]['test']['images'] = (datasets[key]['test']['images'].astype('float32')  - image_mean)/255.
    elif t is 'allnorm':
        for key in datasets.keys():
            
            if key =='mnist' or key == 'usps':
                tmp_1 = datasets[key]['train']['images'][:(len(datasets[key]['train']['images']) // 2)]
                tmp_2 = datasets[key]['train']['images'][(len(datasets[key]['train']['images']) // 2):]
                datasets[key]['train']['images'] = np.concatenate([Normalize(tmp_1),Normalize(tmp_2)])
                datasets[key]['valid']['images'] = Normalize(datasets[key]['valid']['images'])
                datasets[key]['test']['images'] = Normalize(datasets[key]['test']['images'])
            else:
                cuts = [datasets[key]['train']['images'].shape[0], 
                datasets[key]['valid']['images'].shape[0],
                datasets[key]['test']['images'].shape[0]]
                all_data = np.concatenate([datasets[key]['train']['images'], datasets[key]['valid']['images'], datasets[key]['test']['images']], 0)
                all_data = Normalize(all_data)
                datasets[key]['train']['images'] = all_data[:cuts[0]]
                datasets[key]['valid']['images'] = all_data[cuts[0]:cuts[0]+cuts[1]]
                datasets[key]['test']['images'] = all_data[cuts[0] + cuts[1]:]

    elif t is 'norm':
        for key in datasets.keys():
            if key =='mnist':
                tmp_1 = datasets[key]['train']['images'][:(len(datasets[key]['train']['images']) // 2)]
                tmp_2 = datasets[key]['train']['images'][(len(datasets[key]['train']['images']) // 2):]
                datasets[key]['train']['images'] = np.concatenate([Normalize(tmp_1),Normalize(tmp_2)])
            else:
                datasets[key]['train']['images'] = Normalize(datasets[key]['train']['images'])
            
            datasets[key]['valid']['images'] = Normalize(datasets[key]['valid']['images'])
            datasets[key]['test']['images'] = Normalize(datasets[key]['test']['images'])


    return datasets




def SelectDataByLabel(datasets,train_label_ratio, test_label_ratio, sources, targets, labelidx = None):  
    num_labels = len(test_label_ratio)
    if labelidx is None:
        labelidx = np.asarray(range(num_labels)).reshape(-1)
    for key in datasets.keys():
        if sources.has_key(key):
            ratio = train_label_ratio[ sources[key]]
        else:
            ratio = test_label_ratio
        ratio = np.asarray(ratio)
        for d_set in datasets[key].keys():
            labels = np.argmax(datasets[key][d_set]['labels'], axis = 1)
            ids = np.asarray(range(len(labels)))
            counts = []
            for i in range(num_labels):
                l = labelidx[i]
                counts.append(len(ids[labels == l])) 
            ## Compute label number
            for i in range(num_labels):
                if ratio[i] == 0:
                    continue
                temp_ratio = ratio/ratio[i] 
                N = counts[i]
                select_nums = N * temp_ratio
                temp = counts - select_nums
                if all(i >= 0 for i in temp):
                    break 
            select_nums = np.floor(select_nums)
            selected_ids = []
            for i in range(num_labels):
                l = labelidx[i]
                n =int(select_nums[i])
                ids = np.asarray(range(len(labels)))
                ids = ids[labels == l]
                shuffle = np.random.permutation(len(ids))
                shuffle = shuffle[:n]
                selected_ids.append(ids[shuffle])
            selected_ids = np.concatenate(selected_ids, axis = 0)
            selected_ids = selected_ids[np.random.permutation(len(selected_ids))]
            datasets[key][d_set]['images'] = datasets[key][d_set]['images'][selected_ids]
            temp = datasets[key][d_set]['labels'][selected_ids]
            datasets[key][d_set]['labels'] = temp[:,labelidx]
    return datasets

def GetLabelRatio(datasets):
    counts = {}
    weights = {}
    for key in datasets.keys():
        counts[key] = {}
        weights[key] = {}
        for d_set in datasets[key].keys():
            temp = np.float32(np.sum(datasets[key][d_set]['labels'], axis = 0))
            counts[key][d_set] = temp/np.sum(temp)
            ratio = np.zeros(counts[key][d_set].shape)
            for i in range(len(counts[key][d_set])):
                if counts[key][d_set][i] > 0:    
                    ratio[i] = 1.0/counts[key][d_set][i]
                else:
                    ratio[i] = 0
            ratio = ratio/np.linalg.norm(ratio, ord = 1)
            weights[key][d_set] = ratio
    return counts, weights

def GetDataCenter(features, labels, domains):   
    L = labels.shape[1]
    avg_all = np.zeros((L,features.shape[1]))
    centers = np.zeros((domains.shape[1] - 1, L, features.shape[1]))
    for l in range(L):
        idxs = labels[:,l] == 1
        avg_all[l] = np.mean(features[idxs], axis = 0)
    for i in range(domains.shape[1] - 1):
        idxs = domains[:,i] == 1
        temp_features = features[idxs]
        temp_labels = labels[idxs]
        centers[i] = np.zeros((L,features.shape[1]))
        for l in range(L):
            l_idxs = temp_labels[:,l] == 1
            if np.sum(l_idxs) == 0:
                centers[i,l,:] = avg_all[l]
            else:
                centers[i,l,:] = np.mean(temp_features[l_idxs], axis = 0)  
    return centers, avg_all
            

#datasets = load_datasets('../Domain/datasets/', sets = {'mnist':1, 'mnistm':1})
#sources = {'mnist':1}
#targets = {'mnistm':1}
#train_label_ratio = [0.01,0.01,0.01,0.01,0.01,0.19,0.19,0.19,0.19,0.19]
#test_label_ratio = [0.19,0.19,0.19,0.19,0.19,0.01,0.01,0.01,0.01,0.01]
#datasets = select_data_byLabel(datasets, train_label_ratio, test_label_ratio, sources, targets)
#counts = get_label_ratio(datasets)







