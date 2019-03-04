import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import csv
from scipy.ndimage import rotate
from scipy.misc import imread, imshow
from tensorflow.examples.tutorials.mnist import input_data

r = csv.reader(open('female.csv','r'),delimiter=',')
r =  list(r)
r = np.array(r)
lab =  r.astype(np.float)

r1 = csv.reader(open('mix.csv','r'),delimiter=',')
r1 =  list(r1)
r1 = np.array(r1)
data =  r1.astype(np.float)
mb_size = 64
z_dim = 64 #latent dim
X_dim = 513 #input dim
y_dim = 513 #label dim
h_dim = 128 #encoder layer dim
c = 0
lr = 1e-8
noise_factor = 0.09
model_path = "female_tmp/model.ckpt"


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(513), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])
c = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

#Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_W1 = tf.get_variable("weight_mr",[X_dim, h_dim],dtype =  tf.float32, initializer = tf.random_normal_initializer(0,0.01))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

#Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_W2_mu = tf.get_variable("weight_mu",[h_dim, z_dim],dtype =  tf.float32, initializer = tf.random_normal_initializer(0,0.01))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

#Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_W2_sigma = tf.get_variable("weight_sig",[h_dim, z_dim],dtype =  tf.float32, initializer = tf.random_normal_initializer(0,0.01))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


""" P(X|z) """
#P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_W1 = tf.get_variable("weight_de",[z_dim, h_dim],dtype =  tf.float32, initializer = tf.random_normal_initializer(0,0.01))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

#P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_W2 = tf.get_variable("weight1_de",[h_dim, X_dim],dtype =  tf.float32, initializer = tf.random_normal_initializer(0,0.01))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

P_W3 = tf.get_variable("weight1_de1",[h_dim, X_dim],dtype =  tf.float32, initializer = tf.random_normal_initializer(0,0.01))
P_b3 = tf.Variable(tf.zeros(shape=[X_dim]))

P_W4 = tf.get_variable("weight1_de2",[X_dim, X_dim],dtype =  tf.float32, initializer = tf.random_normal_initializer(0,0.01))
P_b4 = tf.Variable(tf.zeros(shape=[X_dim]))



def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    X_mu= tf.matmul(h, P_W2) + P_b2
    X_log_var=tf.matmul(h, P_W3) + P_b3
    return X_mu,X_log_var

def R(z):
    logits = tf.matmul(X, P_W4) + P_b4
    prob = tf.nn.sigmoid(logits)
    return prob, logits,

""" Training """

z_mu, z_logvar = Q(c)#Projection to latent space
z_sample = sample_z(z_mu, z_logvar) #generate samples from latent space
#softsam =  tf.nn.softmax(z_sample, dim=0)
X_mu,X_log_var = P(z_sample) #output from decoder
#z_sample1 = sample_z(X_mu, X_log_var)
#_, logits = R(z_sample1)
recon_loss = tf.reduce_mean(tf.div(tf.square(X_mu - X),tf.exp(X_log_var)))
#recon_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = X)
#recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X_noise) || P(z|X)); calculate in closed form as both dist. are Gaussian
#kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
latent_var = 0.5
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar/latent_var) + z_mu**2/latent_var - (1.- tf.log(latent_var)) - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)
solver = tf.train.AdamOptimizer().minimize(vae_loss)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())



i = 0
n_epochs=1000
train = []

for epoch in range(n_epochs):
    label = []
    for it in range(17): #17 is #frame
        ipt1 = lab[(it*13):((it+1)*13),:]
        ipt1 = np.reshape(ipt1,(13,513)) # read input data
        ipt2 = lab[(it*13):((it+1)*13),:]
        ipt2 = np.reshape(ipt2,(13,513)) #read labels
        #print ipt2
        _,loss,mu,sig = sess.run([solver,vae_loss,X_mu, X_log_var], feed_dict={c: ipt1,X: ipt2})#input to network
        if epoch ==999:
            for i in range(13):
    	        f = open( 'mu_female_train.csv', 'a' )
            	w = csv.writer(f,delimiter = ',', quoting= csv.QUOTE_MINIMAL)
            	w.writerow(mu[i])
#           	f.write( 'dict = ' + repr(samples) + '\n' )
            	f.close()
            for i in range(13):
    	        f = open( 'sig_female_train.csv', 'a' )
            	w = csv.writer(f,delimiter = ',', quoting= csv.QUOTE_MINIMAL)
            	w.writerow(sig[i])
#           	f.write( 'dict = ' + repr(samples) + '\n' )
            	f.close()
    print loss
    train.append(loss)

save_path = saver.save(sess, model_path)   
print("Model saved in file: %s" % model_path)             
train = np.array(train)
xe = np.array(range(n_epochs))
lines = plt.plot(xe,train,'b')
#lines1 = plt.plot(x,val,'r')
#plt.tight_layout()
plt.autoscale(enable=True, axis='x', tight=True)
plt.setp(lines, linewidth = 2.0)
#plt.setp(lines1, linewidth = 2.0)
plt.ylabel('Loss', fontweight='bold')
plt.xlabel('No. of Epochs', fontweight='bold')
plt.show()
