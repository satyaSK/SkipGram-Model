# Skip-Gram Model 
* The above code implements the Skip-gram model from [this paper](https://arxiv.org/pdf/1310.4546.pdf). The main idea is to form word vector representations without losing the semantic information and relationships amongst various words.
* This is achieved by mapping the words to a higher dimensional space. As a final observation it seems that similarities between words seem to be encoded in the vector difference between the words. 
* I've also included a word embedding visualization to visualize the output of this project.

## Dependencies
* Tensorflow
* Numpy
* tqdm
Install dependencies using
```
pip install <package name>
```

## Pipeline
```
Words -> Convert to vector embeddings -> feed into the model -> calculate loss -> backprpagate error -> update vector embeddings
```

## Word Embeddings
![wordembedding](https://user-images.githubusercontent.com/34591573/34705676-24fea6ac-f528-11e7-9cdd-4a45a6560d05.png)

* What you see, are the word vector embeddings, with the blue spheres representing indivisual words reduced down from 128-D space to 3-D space for visualization
* Notice how they have begun to form clusters, with semantically similar words(occuring in the same context) coming closer together and dissimilar words going far away from each other. 
* I haven't incuded the word tags in this picture to keep the visualization clean. But they can be easily seen on tensorboard.

## What are word Embeddings bro?
* As defined by [Christopher Olah](http://colah.github.io), word embedding is a parameterized function whaich maps words in some given language, to higher dimensional vectors. And this fucntion is defined by the ```embedding matrix```.
* Our network cannot and does not deal with strings during training, for this reason, before we pass the words through our network, we have to convert them into numbers of some form(not just plain numbers).
* Converting the words into plain integers to pass them for training is of no use for 2 reasons
	* The integer associated with each word will vary in magnitude(in a random way, rendering it semantically senseless).
	* Choosing single integers for indivisual word would not capture the relationships amongst words having similar meaning, as there is no way to encode the similarity within the embeddings as for a given word vector, there will be a single ```1``` and the rest of the values in that vector will be ```0```.
* Our second option is to use one-hot encodings to represent inivisual words. But this approach fails to properly represent the words occuring in similar contexts.
* So finally, we arrive at word vector embeddings. These vector embeddings are multi-dimensional representations of indivisual words. To make it less scary, word embeddings are simple statistical representations of words, in which a single word can be represented by a single vector, for example, a vector(or embedding) of size 128(aka 128-D embedding).
* We use vector embeddings for representing words as they are really good at capturing relationships between similar words or words occuring in similar contexts. And they are surprisingly good at capturing relationships between words, which wouldn't occur to humans normally.
* So we eventually want to train this vector representation aka ```embedding_matrix``` of words to represent words and relationships between words more accurately, such that words meaning the same thing or occuring in the same context can be found closer to each other in multi-dimensional space.
* Here is a mode in-depth dissection of the [essence of word vector embeddings](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/).

## Dataflow Graph
![skipgram](https://user-images.githubusercontent.com/34591573/34705348-d91e0a90-f525-11e7-9eac-a946637f078a.png)
A simple abstract visualization of the skip Gram model.

## What's a Skip-Gram Model bro?
* By this time we have the understanding of word embeddings on point. Now, we want to train our Word2Vec model to produce good vector representations for words.
* There are some issues with "standard" neural networks, when it comes to learning word vectors.
* There are 2 primary models which help us solve this and achieve our target:
	* The continuous bag of words model(aka CBOW)
	* The skip gram model(here i'll be focusing on this model)
* Given a huge corpus of text, which is the converted to tokens(words), the basic job of the skip gram model is to predict a context word(target word), given a center word.
* Here's an example to explain this better. Consider the sentence "I love to eat food", for this sentence, given the center word is "love", we can safely say that the words "I", "to", "eat" and "food" are context words, which occur in the same "context" as the word "love".
* So, we want to train a model to get good at finding vector representations of context words given the center words.
* As the skip gram model learns embeddings for words, similar words tend to have similar embeddings. Also these similar words are seperated by smaller distances in multi-dimensional space.

## An Interesting Observation! 
* Notice that all the words in the embedding matrix of size ```vocab_size``` will have, not one, but two vector representations.
	* Firstly, when that word is a center word.
	* Secondly, when the word is a occurs as a context word.
* This inherent nature of the model, adds up to the accuracy with which it performs.

## Data processing
* The data processing takes Hardwork! Trust me!
* Firslty we download the dataset to ```downloaded_dataset_path```, if not already present. We unzip its contents and retrieve the file and return the ```tokens```.
* The dataset we are dealing with is the text8 Wikipedia dump. It contains 253,855 unique words and the total word count is 17,005,208.
* The ```most_common``` words are given unique integers ID's. Therefore, each unique word would be associated with an integer which would, in turn, be associated with it's vector embedding of size ```word_embedding_size```. Also the "not so common" words are replaced by the token ```NAN```.
* The dictionary of word ID's and the words themselves are passed to the ```generate_pairs``` function which generates center-target word pairs within a window of fixed size. This is one of the hyperparameters.
* Finally, the ```next-batch``` function is created to fetch the next batch to pass into our model. It will fetch integers corresponding to unique words.
* Personally for me, understanding and implementing the data preprocessing part took up most of the time, but now, once this heavy-lifting is done, its just a matter of creating the model!

## Simplified Approach
* The model will be taking in words(i.e. integers) as its inputs. Also an important hyperparameter ```neg_samples = 64``` is defined which is the number of negative(aka incorrect) samples which our model will be trained on. This is done to increase the seperation between dissimilar words in multi-dimensional space, when they are passed into the loss function. This is called negative sampling.
* These integers should now be converted to vector representations, this would be done by looking up into the embedding matrix, which is a humongous matrix of size ```vocab_size x word_embedding_size``` which holds the representation of each word in the vocabulary.
* Though this lookup is somehwhat easy to carry out(as it's only a matter of selecting columns which could be done by one-hot encoders), Tensorflow provides a handy ```tf.nn.embedding_lookup``` function which easily gets the vector embeddings(initially untrained) for the required words.     
* Once this is done, the weights and biases for the Noise contrastive estimation(NCE) loss are defined and passed into Tesorflow's ```tf.nn.nce_loss``` function.
* Now the model is trained, the loss is calculated, and the vector representations of every word are updated in the right direction to minimize the objective function.

## Noise Contrastive Loss 
![loss](https://user-images.githubusercontent.com/34591573/34705464-9cb6dc16-f526-11e7-841d-aeab880103ac.png)

* Observe how the loss plummets and saturates when it reaches 70k iteration mark. I've used the ```tf.train.GradientDescentOptimizer``` to optimize this loss function, and smoothened the results a little, for observing the trend.
* Here is some more, in-depth analysis of the [Noise Contrastive Loss](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf).

## Objective function
* Getting a probablity distribution of words over the entire corpus via the softmax function is a very computationally expensive task.
* To get around this, we use what is called the noise contrastive estimation loss, which treats a multinomial classification problem as a simple binary logistic regression problem.
* So for each training example, the loss fucntion is fed in a correct center-target pair and ```neg_sample``` number of incorrect pairs, where the center word is kept constant and the target words are randomly chosen incorrect words.
* By learning to distinguish between the correct pairs and the incorrect one's, the embedding matrix eventually "learns" vector embeddings.
* What we perceive is that the correct context word are being learnt, which is actually a product of the model being able to distinguish, for a given center word, whether another given word is a good word or a bad word, i.e. the model asks the question "Can this target word occur in the context of the center word?"
* The true beauty of neural networks can be observed with the Word2Vec model, where it learns true vector representations of elements, which take care of semantic similarity

## A word on Visualization
* When we are dealing with models like the Word2Vec, it is important to get an idea of what the data looks like.
* The words are higher dimensional vector representations(in this case 128 dimensional), so in order to understand them intuitively, how could humans visualize these?
* The answer lies within dimensionality reduction techniques like PCA(pricipal component analysis) and t-SNE(t-Distributed Stochastic neighbor embedding)
* Here, the visualization which I produced at the top of the page, uses t-SNE, which unlike PCA, is a non-linear dimensionality reduction algorithm which maps higher dimensional vector embeddings to lower dimensional one's for human visualization.
* Though i'm not going to go deep into the detailed functioning of the algorithm, its a pretty sweet visualization tool built into tensorboard which helps in exploring the data and visualizing our efforts :D
* More on t-SNE can be found on the papers written by [Geoffrey Hinton](http://papers.nips.cc/paper/2276-stochastic-neighbor-embedding.pdf) and [Laurens van der Maaten](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).

## Basic Usage
For Running, type in terminal
```
python skipGramModel.py
```
For the word Embedding visualization, type in terminal
```
tensorboard --logdir="Visualize_Embeddings"
```
For the beautiful dataflow graph visualization, type in terminal
```
tensorboard --logdir="visualize"
```



