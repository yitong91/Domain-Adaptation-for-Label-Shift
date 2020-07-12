import tensorflow as tf
import utils
import numpy as np
import tensorflow.contrib.layers as layers
#flip_gradient = FlipGradientBuilder()
class DATS(object):
    """Simple MNIST domain adaptation model."""
    def __init__(self, options):
        self.l = tf.placeholder(tf.float32, [])
        self.lr_g = tf.placeholder(tf.float32, [])
        self.lr_d = tf.placeholder(tf.float32, [])
        self.lr_lp = tf.placeholder(tf.float32, [])
        self.sample_type = tf.float32
        self.ef_dim = options['ef_dim']
        self.num_labels = options['num_labels']
        self.latent_dim = options['latent_dim']
        self.num_domains = options['num_domains']
        self.sample_shape = options['sample_shape']
        self.num_source = options['num_source']
        self.num_target = options['num_target']
        self.X = tf.placeholder(tf.as_dtype(self.sample_type), [None] + list(self.sample_shape), name="input_X")
        self.y = tf.placeholder(tf.float32, [None, self.num_labels], name="input_labels")
        self.domains = tf.placeholder(tf.float32, [None, self.num_domains], name="input_domains")
        self.train = tf.placeholder(tf.bool, [], name = 'train')
        self.label_weight = tf.placeholder(tf.float32, [None,self.num_labels], name = "label_weight")
        
        self.domain_weight = tf.placeholder(tf.float32, [self.num_domains,], name = "domain_weight")
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.test_label_ratio = tf.get_variable('test_label_weight', shape = [self.num_labels,],
                                                initializer=tf.constant_initializer(1.0/self.num_labels))
        self.mus = tf.placeholder(tf.float32, [self.num_domains-1, self.num_labels, self.latent_dim])
        self.target_mu = tf.placeholder(tf.float32, [self.latent_dim,])
        self._build_model()
        self._setup_train_ops()
        
    def GetDomainWeights(self, d, pred = None):
        if pred is None:
            return np.ones((self.num_domains,), dtype = np.float32)
        domain_weights = np.zeros(pred.shape)
        for i in range(self.num_domains):
            domain_weights[:,i] = pred[:,-1] * d[:,i]
        domain_weights[:,-1] = 0
        domain_weights = np.sum(domain_weights, axis = 0)  
        domain_sample_num = np.sum(d, axis = 0)
        for i in range(self.num_domains):
            if domain_sample_num[i] == 0:
                domain_sample_num[i] = 1
        domain_weights = domain_weights/domain_sample_num
        domain_weights = domain_weights
        d_norm = np.linalg.norm(domain_weights[:-1], ord = 1)
        if  d_norm <1e-5:
            domain_weights = (1.0/(self.num_domains-1)) * np.ones((self.num_domains,), dtype = np.float32)
        else:
            domain_weights = domain_weights/d_norm
            #domain_weights = utils.softmax(domain_weights)
        domain_weights[-1] = 1
        return domain_weights.astype('float32')    
        
    
    def GetMus(self, features, y, d):
        source_features = features[d[:,-1] != 1]
        target_features = features[d[:,-1] == 1]
        d = d[d[:,-1] != 1]
        target_mu = np.mean(target_features, 0)
        source_mus = np.zeros([self.num_domains-1, self.num_labels, self.latent_dim])
        for i in range(self.num_domains - 1):
            for j in range(self.num_labels):
                index = d[:,i] * y[:,j]
                selected_features = source_features[index == 1]
                source_mus[i, j, :] = np.mean(selected_features, 0) 
        return source_mus, target_mu
    
    def LabelProportionLoss(self):
        test_label_ratio = tf.expand_dims(tf.nn.softmax(self.test_label_ratio), 1)
        lp_loss = 0
        for d in range(self.num_domains - 1):
            s_mu = tf.reduce_sum(test_label_ratio * self.mus[d], 0)
            lp_loss += self.domain_weight[d] * tf.reduce_mean(tf.square(s_mu - self.target_mu))
        self.test1 = test_label_ratio
        self.test2 = s_mu
        return lp_loss
        
    def FeatureExtractor(self, X, reuse = False):
        input_X = utils.NormalizeImage(X)
        with tf.variable_scope('feature_extractor_conv1',reuse = reuse):
            h_conv1 = layers.conv2d(input_X, self.ef_dim, 3, stride=1,
                                    activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            h_conv1 = layers.conv2d(h_conv1, self.ef_dim, 3, stride=1,
                                    activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            h_conv1 = layers.max_pool2d(h_conv1, [2, 2], 2, padding='SAME')        
            
        with tf.variable_scope('feature_extractor_conv2',reuse = reuse):  
            h_conv2 = layers.conv2d(h_conv1, self.ef_dim * 2, 3, stride=1,
                                    activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            h_conv2 = layers.conv2d(h_conv2, self.ef_dim * 2, 3, stride=1,
                                    activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            h_conv2 = layers.max_pool2d(h_conv2, [2, 2], 2, padding='SAME')
            
        with tf.variable_scope('feature_extractor_conv3',reuse = reuse):  
            h_conv3 = layers.conv2d(h_conv2, self.ef_dim * 4, 3, stride=1,
                                    activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            h_conv3 = layers.conv2d(h_conv3, self.ef_dim * 4, 3, stride=1,
                                    activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            h_conv3 = layers.max_pool2d(h_conv3, [2, 2], 2, padding='SAME')
            
        with tf.variable_scope('feature_extractor_fc1', reuse = reuse):
            fc_input = layers.flatten(h_conv3)
            fc_1 = layers.fully_connected(inputs=fc_input, num_outputs=self.latent_dim,
                                              activation_fn=None, weights_initializer=self.initializer)
            features =  fc_1
        return features
            
    def LabelPredictor(self, features, reuse = False):
        h_fc1_for_prediction =  tf.cond(self.train,  lambda: tf.slice(features, [0, 0], [self.num_source, -1]), lambda: features)           
        with tf.variable_scope('label_predictor_logits', reuse = reuse):
            logits = layers.fully_connected(inputs=h_fc1_for_prediction, num_outputs=self.num_labels,
                                              activation_fn=None, weights_initializer=self.initializer)   
        y_pred = tf.nn.softmax(logits)
        y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.y))
        y_acc = utils.PredictorAccuracy(y_pred,self.y)
        return y_pred, y_loss, y_acc
        

    def DomainPredictor(self, features,  reuse = False):
        # Flip the gradient when backpropagating through this operation
        with tf.variable_scope('domain_predictor_fc1', reuse = reuse):
            d_h_fc0 = layers.fully_connected(inputs=features, num_outputs=100,
                                              activation_fn=tf.nn.relu, weights_initializer=self.initializer) 
        with tf.variable_scope('domain_predictor_logits', reuse = reuse):
            d_logits = layers.fully_connected(inputs=d_h_fc0, num_outputs=self.num_domains,
                                              activation_fn=None, weights_initializer=self.initializer) 
        d_pred = tf.nn.softmax(d_logits)
        ##
        add_on = np.ones([self.num_source + self.num_target])
        add_on[:self.num_source] = 0.0
        add_on =  tf.constant(add_on, tf.float32)
        self.add_on = add_on
        weight = (1.0/self.label_weight) * tf.expand_dims(tf.nn.softmax(self.test_label_ratio), axis = 0)    
        weight = tf.reduce_sum(weight,axis = 1) + add_on
        
        self.weight = tf.clip_by_value(weight, 0.1,2.0)
        d_loss = tf.reduce_mean(self.weight * tf.nn.softmax_cross_entropy_with_logits(logits = d_logits, labels = self.domains))
        d_acc = utils.PredictorAccuracy(d_pred,self.domains)
        return d_pred, d_loss, d_acc
    
    
    
    def _build_model(self):     
        
        self.features = self.FeatureExtractor(self.X)       
        self.y_pred, self.y_loss, self.y_acc = self.LabelPredictor(self.features)
        self.d_pred, self.d_loss, self.d_acc = self.DomainPredictor(self.features)
        self.total_loss = self.y_loss - self.l*self.d_loss
        self.lp_loss = self.LabelProportionLoss() 
        
    def _setup_train_ops(self):
        label_vars = utils.VarsFromScope(['label_predictor', 'feature_extractor'])
        domain_vars = utils.VarsFromScope(['domain_predictor'])
        lp_vars = utils.VarsFromScope(['test_label_weight'])
        self.train_feature_ops = tf.train.AdamOptimizer(self.lr_g, 0.5).minimize(self.total_loss, var_list = label_vars)
        self.train_adversary_ops = tf.train.AdamOptimizer(self.lr_d, 0.5).minimize(self.l * self.d_loss, var_list = domain_vars)
        self.train_lp_ops =  tf.train.AdamOptimizer(self.lr_lp, 0.5).minimize(self.lp_loss, var_list = lp_vars)
