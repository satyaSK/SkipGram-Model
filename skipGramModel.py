#Import Dependencies
#-----------for backward compatibility of the code-----------
# its the missing link between python2 and python3, so our  you can slowly be accustomed to incompatible changes!
from __future__ import absolute_import, division, print_function 
#------------------------------------------------------------
from tqdm import tqdm
from helperfn import get_data, checkIt, create_directory
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector 
import time
import os

#Check whether data holding folder is present
checkIt()

#Define hyperparameters
batch_size = 128
word_embedding_size = 128
window_size = 1
vocab_size = 50000 #most common 50000 tokens
neg_samples = 64# for negative sampling
learning_rate = 1.0# how fast to let-go of old beliefs?
total_iterations = 10000
skip = 2000

with tf.name_scope('Data'):
	center_words = tf.placeholder(tf.int32, shape=[batch_size], name='center_words')
	target_words = tf.placeholder(tf.int32, shape=[batch_size,1], name='target_words')

#Global step to save ma checkpoints
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

#Embedding matrix
embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, word_embedding_size], -1.0, 1.0), name='embedding_matrix')

def compute_loss():
	with tf.name_scope('Loss'):
		get_embeddings = tf.nn.embedding_lookup(embedding_matrix, center_words, name='embeddings')
		W = tf.Variable(tf.truncated_normal([vocab_size, word_embedding_size], stddev=1.0/(word_embedding_size**0.5)), name='weigths')
		b = tf.Variable(tf.zeros([vocab_size]), name='biases')
		noise_contrastive_loss = tf.nn.nce_loss(weights=W, biases=b, labels=target_words, inputs=get_embeddings, num_sampled=neg_samples, num_classes=vocab_size)
		loss = tf.reduce_mean(noise_contrastive_loss)
		return loss

def optimize(objective):
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(objective, global_step=global_step)
	return optimizer

#generate batches
generated_batch = get_data(vocab_size, batch_size, window_size)
loss = compute_loss()# computing the loss
optimizer = optimize(loss)

#tensorflow ops which outputs protocol buffers containing "summarized" data
with tf.name_scope("summaries"):
	tf.summary.scalar('Loss',loss)	
	summary_op = tf.summary.merge_all()


create_directory('Checkpoints')#make a required directory
saver = tf.train.Saver() #initialize save object


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	graph_writer = tf.summary.FileWriter("./visualize", sess.graph)

	
	ckpt = tf.train.get_checkpoint_state(os.path.dirname('Checkpoints/checkpoint'))
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
	initial_step = global_step.eval()
	

	total_error = 0
	print("\nGood to go! Training Starts\n")
	for i in tqdm(range(initial_step, initial_step + total_iterations)):
		start =time.time()
		center_batch, target_batch = next(generated_batch)
		e,_,summary = sess.run([loss, optimizer, summary_op], feed_dict={center_words:center_batch, target_words:target_batch})
		graph_writer.add_summary(summary, global_step=i)
		total_error += e

		if i%skip == 0:
			end = time.time()
			print('Loss at step {0} is {1:.3f}     Time:{2:.3f}'.format(i, total_error,end-start))
			
			saver.save(sess, 'Checkpoints/checky', i+1)
			
			total_error = 0

	end_embedding_M = sess.run(embedding_matrix)
	Variable_EMBEDDING = tf.Variable(end_embedding_M[:1000], name='top_embedding')# turn embedding matrix into variable
	sess.run(Variable_EMBEDDING.initializer)
	config = projector.ProjectorConfig()
	embedding_writer = tf.summary.FileWriter("Visualize_Embeddings")

	embedding = config.embeddings.add()
	embedding.tensor_name = Variable_EMBEDDING.name
	embedding.metadata_path = "first1000.tsv" #simply the metadata path

	projector.visualize_embeddings(embedding_writer, config)
	saver_embed = tf.train.Saver([Variable_EMBEDDING])
	saver_embed.save(sess, "Visualize_Embeddings/model.ckpt", 1)
	
	#Visualization commands
	#tensorboard --logdir="visualize"
	#tensorboard --logdir="Visualize_Embeddings"
	Name = "visualize"
	Embed_name = "Visualize_Embeddings"
	print("To get dataflow graph, use the command-> tensorboard --logdir=\"" + Name + "\" ")
	print("To visualize Embeddings, Use the command-> tensorboard --logdir=\"" + Embed_name + "\" ")









