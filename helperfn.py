#import Dependencies
#-----------for backward compatibility of the code-----------
# So that it can function in the future also!!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#------------------------------------------------------------
import tensorflow as tf# Good friend tensorflow
from six.moves import urllib # Offcourse we'll be downloading from a given URL
from collections import Counter# for getting number of same tokens
import numpy as np # Coz who doesn't use numpy in deep leanring?
import zipfile # For dealing with downloaded zip files
import random #for randomness
import os #for os related operations
import sys
sys.path.append('..')




download_from_url = 'http://mattmahoney.net/dc/'# download from where??
n_bytes = 31344016# size of the file which will be dowwnloaded
dataset_folder = "Dataset/"#save the dataset to dataset_folder
FILE_name = "text8.zip"#name of the text corpus we want to download
data_holder = "Dataset"
#------------------------------------------------------------------
#Checks whether the data directory is present! If not then it is created!
def checkIt():
	if os.path.exists(data_holder):
		print("Dude, the data folder already exists!!")
		pass
	else:
		create_directory('Dataset')
		print("You didn't have a data folder so it has been added to your working directory!!!")

#------------------------------------------------------------------
#Create a simple directory in the working directory
def create_directory(path):
    try:
        os.mkdir(path)
    except OSError:
    	pass

#-------------------------------------------------------------------
#Download the file if not already downloaded(if already downloaded, return the path)
def download_file(n_bytes, f_name):
	downloaded_dataset_path =  dataset_folder + f_name
	if os.path.exists(downloaded_dataset_path):
		print("Found downloaded dataset")
		return downloaded_dataset_path

	print("Downloading the dataset for you\n")
	f_name, headers = urllib.request.urlretrieve(download_from_url + f_name, downloaded_dataset_path)#returns filename nd headers. Dont need headers!!
	file_info = os.stat(downloaded_dataset_path)
	if file_info.st_size == n_bytes:# or even getsize(fine_name), but internal implementations are the same as st_size(so no difference)
		print("Download Complete!\n" + f_name + "downloaded")
	else:
		raise Exception("Something bad happened, try downloading the file manually from "+download_from_url)
	return downloaded_dataset_path
#--------------------------------------------------------------------
#Read the zip file to get all the tokens
def read_zip(file_path):
	with zipfile.ZipFile(file_path) as f:
		tokens = tf.compat.as_str(f.read(f.namelist()[0])).split()#namelist gives names of file, read() reads the content and tf.compat.as_str converts to strings
	return tokens

#---------------------------------------------------------------------
#Create the dictionary of all the indivisual unique(most common) words
def create_vocab(tokens, vocab_size):
	count = [('NAN', -1)]
	#"extend" and not append the count list, with the most common vocab_size words
	count.extend(Counter(tokens).most_common(vocab_size -1))
	idx = 0
	#create a dictionary for each indivisual unique word.
	dictionary = dict()
	create_directory('Visualize_Embeddings')
	with open('Visualize_Embeddings/first1000.tsv', "w") as f:
		for word, _ in count:
			dictionary[word] = idx
			if idx < 1000:# store the first 1000 words in first1000.tsv
				f.write(word + "\n")  
			idx += 1
		index_dict = dict(zip(dictionary.values(), dictionary.keys()))
	return dictionary, index_dict

#---------------------------------------------------------------------
#Give all the tokens and the dictionary of token to this funtion to spit out all the words converted to indexes
def word_to_idx(words, dictionary):
	word_indexes = []
	unknown = 0
	for word in words:
		if word in dictionary:
			word_indexes.append(dictionary[word])#index of words inside the dictionary
		else:
			word_indexes.append(unknown)#thats the index of all unknown(UNK) words
	return word_indexes

#-----------------------------------------------------------------------
#For every center word--> give back a randomly generated (center_word,target_word) pair, before and after the center word within a radius of radius = window size 
def generate_pairs(word_indexes, window):
	for idx, center_word in enumerate(word_indexes):
		context_idx = random.randint(1,window)
		#backward window of radius=window
		for target_word in word_indexes[max(0, idx - context_idx) : idx]:
			yield center_word, target_word
		#forward window of radius=window
		for target_word in word_indexes[idx + 1 : idx+context_idx+1]:
			yield center_word, target_word

#------------------------------------------------------------------------
#give this function an iteratable(i dont know if that word exits :D) list to generate batches
#IMPORTANT: Im "generating" and not "returning". Baiscally the local variables are not lost in the case of "generating"
def next_batch(iterator, batch_size):
	while True:
		center_batch = np.zeros(batch_size, dtype=np.int32)
		target_batch = np.zeros([batch_size,1])
		for idx in range(batch_size):
			center_batch[idx], target_batch[idx] = next(iterator)
		yield center_batch, target_batch

#------------------------------------------------------------------------
# This function simply returns the dictionary of:
#--converting words to index-- & --converting index to words--
def two_way_dict(vocab_size):
	downloaded_dataset_path = download_file(n_bytes, f_name)
	tokens = read_zip(downloaded_dataset_path)
	return create_vocab(tokens, vocab_size)# returns dictionary and index_dict

#-----------------------------------------------------------------------
#Basically use all the fucntions created so far to generate a simple batch(All this for a simple batch generation?? YES!! It requires a lot of HardWork!)
def get_data(vocab_size, batch_size, window_size):
	downloaded_dataset_path = download_file(n_bytes, FILE_name)#downloaded file
	tokens = read_zip(downloaded_dataset_path)#all the words
	dictionary, _ = create_vocab(tokens, vocab_size)#all the indexes to convert words into indexes
	word_indexes = word_to_idx(tokens, dictionary)# give 'tokens' and 'dictionary' to get back word_indexes
	del tokens #just saving some memory!!
	generated = generate_pairs(word_indexes, window_size)
	batch = next_batch(generated, batch_size)
	return batch

#-----------------------------------------------------------------------
######################### PRE-PROCESSING ENDS #######################
#-----------------------------------------------------------------------






