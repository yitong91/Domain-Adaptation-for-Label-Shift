import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
from model import DATS
import utils
import imageloader as dataloader
import scipy.io
import os
import pdb



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4) 
method = 'DATS'
target_name = 'mnistm'
batch_size = 64
num_steps = 2000
valid_steps = 200
lp_steps = 100

sources = {'mnist':0, 'synth':1,'svhn':2}
targets = {'mnistm':3}
train_label_ratio = [[0.15,0.05,0.15,0.05,0.15,0.05,0.15,0.05,0.15,0.05],
                    [0.05,0.15,0.05,0.15,0.05,0.15,0.05,0.15,0.05,0.15],
                    [0.15,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05]]
test_label_ratio = [0.1, 0.1,0.1,0.1,0.2,0.1,0.05,0.05,0.1,0.1]

datasets = dataloader.LoadDatasets('/home/lyt/Documents/Domain/datasets/',{'mnist':0,'mnistm':0,'svhn':0, 'synth':0})

datasets = dataloader.SelectDataByLabel(datasets, train_label_ratio, test_label_ratio, sources, targets)
datasets = dataloader.NormalizeDataset(datasets)

datasets, source_train, source_valid, source_test, target_train, target_valid, target_test = dataloader.SourceTarget(datasets, sources, targets)

options = {}
options['sample_shape'] = (28,28,3)
options['num_domains'] = len(sources) + len(targets)
options['num_labels'] = 10
options['batch_size'] = batch_size
options['G_iter'] = 1
options['D_iter'] = 1
options['L_iter'] = 100
options['ef_dim'] = 32
options['latent_dim'] = 128
options['num_source'] = batch_size * ( options['num_domains'] - 1)
options['num_target'] = batch_size #* ( options['num_domains'] - 1)
options['l'] = 0.1

ratio_original, ratio = dataloader.GetLabelRatio(datasets)


target_ratio_original = np.ones((options['num_labels'], ))/np.float32(options['num_labels'])
sources_ratio_original = [np.zeros(options['num_labels'])] * len(sources)
for key in sources.keys():
    idx = sources[key]
    sources_ratio_original[idx] = ratio_original[key]['train']
    
    
ratio_train = [[] for i in range(len(sources.keys()))]
for key in sources.keys():
    ratio_train[sources[key]] = ratio[key]['train']


tf.reset_default_graph()  
graph = tf.get_default_graph()
model = DATS(options)
sess =  tf.Session(graph = graph, config=tf.ConfigProto(gpu_options=gpu_options)) 
tf.global_variables_initializer().run(session = sess)

record = []
gen_source_batches = []
for key in sources.keys():
    gen_source_batches.append(utils.BatchGenerator([datasets[key]['train']['images'], datasets[key]['train']['labels'], datasets[key]['train']['domains']], options['batch_size']))
gen_target_batch = utils.BatchGenerator([datasets[targets.keys()[0]]['train']['images'], datasets[targets.keys()[0]]['train']['labels'], datasets[targets.keys()[0]]['train']['domains']], options['num_target'])
save_path = './Result/DATS_' +  targets.keys()[0] + '/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


best_valid = -1
best_acc = -1
d_pred = None

print('Training...')
source_features = utils.GetDataPred(sess, model, 'feature', source_train['images'])
target_features = utils.GetDataPred(sess, model, 'feature', target_test['images'])
source_mus, target_mu = model.GetMus(np.concatenate([source_features, target_features], 0),
                                    source_train['labels'],
                                    np.concatenate([source_train['domains'], target_test['domains']], 0))

for i in range(1, num_steps + 1):          
    # Adaptation param and learning rate schedule as described in the paper
    p = float(i) / num_steps
    l = options['l'] #* (2. / (1. + np.exp(-10. * p)) - 1)
    lr_g = 0.001 #/ (1. + 10 * p)**0.75
    lr_d = 0.001 #/ (1. + 10 * p)**0.75
    lr_lp = 0.001#/ (1. + 10 * p)**0.75
    #l = 0.005
    #lr = 0.0001
    X0 = []
    y0 = []
    d0 = []
    for j in range(len(sources.keys())):
        x_temp, y_temp, d_temp = gen_source_batches[j].next()
        X0.append(x_temp)
        y0.append(y_temp)
        d0.append(d_temp)
    X0 = np.concatenate(X0, axis = 0)
    y0 = np.concatenate(y0, axis = 0)
    d0 = np.concatenate(d0, axis = 0)
    X1, _, d1 = gen_target_batch.next()
    X = np.concatenate([X0, X1], axis = 0)
    d = np.concatenate([d0, d1], axis = 0)
    domain_weight = model.GetDomainWeights(d, d_pred)
    source_labelweight = utils.GetLabelWeight(sources_ratio_original, np.ones((options['num_labels'],)), y0, d0, domain_weight)
    label_weight = np.concatenate(( np.expand_dims(source_labelweight,axis = 1) * y0, 
                                   np.zeros((options['num_target'], options['num_labels']))), axis = 0)

    for j in range(options['G_iter']):
        
        # Update Feature Extractor & Lable Predictor np.array([0,0,1,1]).astype('float32')
        _, d_pred, batch_loss, features,  tploss, td_acc, tp_acc = \
            sess.run([model.train_feature_ops, model.d_pred, model.total_loss, model.features, model.y_loss, model.d_loss, model.y_acc],
                     feed_dict={model.X: X, model.y: y0, 
                                model.domains: d, model.l: l, 
                                model.lr_g: lr_g, 
                                model.label_weight:label_weight, 
                                model.train: True})

    
    
    for j in range(options['D_iter']):
        # Update Adversary
        _, d_pred, d_acc = \
            sess.run([model.train_adversary_ops, model.d_pred, model.d_acc],
                     feed_dict={model.X:X, model.y:y0, model.domains: d,  
                     model.l: l, model.lr_d: lr_d, 
                     model.label_weight:label_weight, 
                     model.mus: source_mus,
                     model.target_mu: target_mu,
                     model.train: True})
     
    if i%lp_steps == 0 or i <=1:
        #Compute target label proportion
        source_features = utils.GetDataPred(sess, model, 'feature', source_train['images'])
        target_features = utils.GetDataPred(sess, model, 'feature', target_test['images'])
        source_mus, target_mu = model.GetMus(np.concatenate([source_features, target_features], 0),
                                            source_train['labels'],
                                            np.concatenate([source_train['domains'], target_test['domains']], 0))
        for j in range(options['L_iter']):
            _, lp_loss = sess.run([model.train_lp_ops, 
                                  model.lp_loss], 
                                feed_dict = {model.domain_weight: domain_weight, 
                                model.label_weight:label_weight,
                                model.target_mu: target_mu,
                                model.lr_lp: lr_lp, 
                                model.train: True, 
                                model.mus:source_mus})


    if i % 100 == 0:
        print '%s iter %d  loss: %.4f  d_acc: %.4f  p_acc: %.4f  lp: %.4f  ' % \
                ('DATS_' + targets.keys()[0], i, batch_loss, d_acc, tp_acc, lp_loss)
        
        
    if i % valid_steps == 0:
        

        source_train_pred = utils.GetDataPred(sess, model, 'y', source_train['images'], source_train['labels'])
        source_train_acc = utils.GetAcc(source_train_pred, source_train['labels'])
        features = utils.GetDataPred(sess, model,'feature',source_train['images'], source_train['labels'])
       
        source_valid_pred = utils.GetDataPred(sess, model, 'y', source_valid['images'], source_valid['labels'])
        source_valid_dpred = utils.GetDataPred(sess, model, 'd', source_valid['images'], source_valid['domains'])
        source_valid_acc = utils.GetAcc(source_valid_pred, source_valid['labels'])
        
        target_test_pred = utils.GetDataPred(sess, model, 'y', target_test['images'], target_test['labels'])
        target_test_dpred = utils.GetDataPred(sess, model, 'd', target_test['images'], target_test['domains'])
        target_test_acc = utils.GetAcc(target_test_pred, target_test['labels'])
        
        domain_acc = utils.GetAcc(np.concatenate((source_valid_dpred, target_test_dpred), axis = 0), np.concatenate((source_valid['domains'], target_test['domains']),axis = 0))
        if source_valid_acc > best_valid:
                best_valid = source_valid_acc
                best_acc = target_test_acc
                confusion = metrics.confusion_matrix(np.argmax(target_test['labels'],axis = 1),np.argmax(target_test_pred,axis = 1))  
        labd = np.concatenate((source_valid['domains'], target_test['domains']), axis = 0)
        print 'train: %.4f  valid: %.4f  test: %.4f domain: %.4f' % \
                (source_train_acc, source_valid_acc, target_test_acc, domain_acc)        
    
print('Best target accuracy is ' + str(best_acc))
#scipy.io.savemat('result.mat',{'record':record,'confusion':confusion})