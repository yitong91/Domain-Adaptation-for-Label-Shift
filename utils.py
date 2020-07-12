import tensorflow as tf
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy
from collections import OrderedDict
# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts

def AssignSampleWeight(ratio, y0, d0, domain_weights=None):
    if domain_weights is None:
        domain_weights = np.ones((d0.shape[1],))
    N = y0.shape[0]
    weight = np.zeros((N,))
    domain = np.argmax(d0, axis = 1)    
    label = np.argmax(y0,axis = 1)
    for i in range(N):
        weight[i] = domain_weights[domain[i]] * ratio[domain[i]][label[i]] * np.float32(len(ratio))
    return weight
    
    
def GetLabelWeight(sources_ratio, target_ratio, y0, d0, domain_weights = None):
    label_ratio = [np.zeros(y0.shape[1],)] * len(sources_ratio)
    for i in range(len(sources_ratio)):
        temp = np.zeros((y0.shape[1],))
        for j in range(len(sources_ratio[i])):
            if target_ratio[j] == 0:
                continue
            temp[j] = sources_ratio[i][j]/target_ratio[j]
        label_ratio[i] = Ratio2Weight(temp)
    weights = AssignSampleWeight(label_ratio, y0, d0, domain_weights)
    return weights
    
    
def GetParams(sess):
    variables = tf.trainable_variables()
    params = {}
    for i in range(len(variables)):
        name = variables[i].name
        params[name] = sess.run(variables[i])
    return params
    
    
def ToOneHot(x, N = -1):
    x = x.astype('int32')
    if np.min(x) !=0 and N == -1:
        x = x - np.min(x)
    x = x.reshape(-1)
    if N == -1:
        N = np.max(x) + 1
    label = np.zeros((x.shape[0],N))
    idx = range(x.shape[0])
    label[idx,x] = 1
    return label.astype('float32')
    
def ImageMean(x):
    x_mean = x.mean((0, 1, 2))
    return x_mean

def VarsFromScope(scopes):
    """
    Returns list of all variables from all listed scopes. Operates within the current scope,
    so if current scope is "scope1", then passing in ["weights", "biases"] will find
    all variables in scopes "scope1/weights" and "scope1/biases".
    """
    current_scope = tf.get_variable_scope().name
    #print(current_scope)
    if current_scope != '':
        scopes = [current_scope + '/' + scope for scope in scopes]
    var = []
    for scope in scopes:
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            var.append(v)
    return var


def ShuffleAlignedList(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

def NormalizeImage(img_batch):
    fl = tf.cast(img_batch, tf.float32)
    return tf.map_fn(tf.image.per_image_standardization, fl)


def BatchGenerator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = ShuffleAlignedList(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = ShuffleAlignedList(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
        
    
    

def PredictorAccuracy(predictions, labels):
    """
    Returns a number in [0, 1] indicating the percentage of `labels` predicted
    correctly (i.e., assigned max logit) by `predictions`.
    """
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)),tf.float32))

def GetDataPred(sess, model, obj_acc, data, labels = None, batch = 1024):
    N = data.shape[0]
    n = np.ceil(N/batch).astype(np.int32)
    if obj_acc == 'feature':
        temp = sess.run(model.features,feed_dict={model.X: data[0:2].astype('float32'), model.train: False})
        pred = np.zeros((data.shape[0],temp.shape[1])).astype('float32')
    else:
        pred= np.zeros(labels.shape).astype('float32')
    srt = 0
    edn = 0
    for i in range(n + 1):
        srt = edn
        edn = min(N, srt + batch - 1)
        X = data[srt:edn]
        if obj_acc is 'y':
            pred[srt:edn,:] = sess.run(model.y_pred,feed_dict={model.X: X.astype('float32'), model.train: False})
        elif obj_acc is 'd':
            pred[srt:edn,:]= sess.run(model.d_pred,feed_dict={model.X: X.astype('float32'), model.train: False})          
        elif obj_acc is 'feature':
            pred[srt:edn] =  sess.run(model.features,feed_dict={model.X: X.astype('float32'), model.train: False})          
    return pred

def GetAcc(pred, label):
    if len(pred.shape) > 1:
        pred = np.argmax(pred,axis = 1)
    if len(label.shape) > 1:
        label = np.argmax(label, axis = 1)
        #pdb.set_trace()
    acc = (pred == label).sum().astype('float32')
    return acc/label.shape[0]


def ImshowGrid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()

def PlotEmbedding(X, y, d, names, title=None, fontsize = 16):
    """Plot an embedding X with the class label y colored by the domain d."""
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    colors = np.array([[0.6,0.4,1.0,1.0],
      [1.0,0.1,1.0,1.0],
      [0.6,1.0,0.6,1.0],
      [0.1,0.4,0.4,1.0],
      [0.4,0.6,0.1,1.0],
      [0.4,0.4,0.4,0.4]]
      )
    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colors[d[i]], 
                 fontdict={'weight': 'bold', 'size': fontsize})

    plt.xticks([]), plt.yticks([])
    patches = []
    for i in range(max(d)+1):
        patches.append( mpatches.Patch(color=colors[i], label=names[i]))
    #plt.legend(handles=patches,prop={'size': 19})
    plt.ylabel('Embedding Dimension 2', fontsize=20, fontweight='bold')
    plt.xlabel('Embedding Dimension 1', fontsize=20, fontweight='bold')
    if title is not None:
        plt.title(title)


def Softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

    
def Sigmoid(x):
  return 1 / (1 + np.exp(-x))


def Ratio2Weight(ratio):
    weight = np.zeros(ratio.shape)
    for i in range(len(ratio)):
        if ratio[i] >0:
            weight[i] = 1.0/ratio[i]
        else:
            continue
    if np.linalg.norm(weight, ord = 1) == 0:
        weight = np.ones(ratio.shape)/np.float32(len(ratio))
    else:
        weight = weight/np.linalg.norm(weight, ord = 1)
    return weight * np.float32(len(ratio))


def SampleShuffle(N, T,sh = True):
    n = np.ceil(float(N)/float(T))
    if sh:
        shuffle = np.random.permutation(N)
    else:
        shuffle = range(N)
    res = []
    for i in range(T):
        srt = int(i*n)
        edn = int(min(srt + n, N))
        res.append(shuffle[srt:edn])
    return res
    