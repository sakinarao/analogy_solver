"""
The purpose of the program is to solve word analogy problems using GloVe word embedding

Source of the code: https://notebook.community/spro/practical-pytorch/glove-word-vectors/glove-word-vectors  
"""


#start of the program
import torch

#Torchtext includes functions to download GloVe (and other) embeddings
import torchtext.vocab as vocab

#loading the pre-trained vectors
glove = vocab.GloVe(name='6B', dim=100)


#prints the number of words loaded
print('Loaded {} words'.format(len(glove.itos)))

def get_word(word):
	return glove.vectors[glove.stoi[word]]
    
def closest(vec, n=10):
    """
    Find the closest words for a given vector
    
    Calculating the cosine distance for each word and sorting based on that distance
    """
    
    cos = torch.nn.CosineSimilarity(dim=0) #To calculate the cosine distance
    
    all_dists = [(w, cos(vec, get_word(w))) for w in glove.itos]
    
    return sorted(all_dists, key=lambda t: t[1], reverse=True)[:n]
    

#Function to print the cosine distance and word pair
def print_tuples(tuples):
    for tuple in tuples:
        print('(%.4f) %s' % (tuple[1], tuple[0]))
        

# In the form w1 : w2 :: w3 : ?
def analogy(w1, w2, w3, n=5, filter_given=True):
    print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))
   
    # w2 - w1 + w3 = w4
    closest_words = closest(get_word(w2) - get_word(w1) + get_word(w3))
    
    # Optionally filter out given words
    if filter_given:
        closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]
        
    print_tuples(closest_words[:n])
    
#Example
analogy('king', 'man', 'queen')


#getting input from the text file 'analogy_problems.txt' and word solving the analogy problems
with open('analogy_problems.txt') as f:
  for i in range(0, 10):
    lines = f.readline().split()
    print()
    analogy(lines[0], lines[1], lines[2])
    
#end of program
    
    
    
"""

1. Which of your 10 cases showed evidence of bias? Show the results of these analogies and explain the nature of the bias. At least 3 of your cases should illustrate some form of bias.

Some of the cases showed biases based on the strereotypes existing in the society.
The following cases showed some form of bias:
a) he : programmer :: she : stylist
The first word closest to she is 'stylist' for this analogy. In this case the model is associating the profession to a specific gender.

b) rich : happy :: poor : tired
In this word analogy, the model is associating the financial staus of an individual with their emotional well being.

c) man : muscular :: woman : skeletal
This is a gender stereotype that a man is muscular and woman is skeletal. This model shows bias based on the gender stereotype.

d) she : nurse :: he : physician
This word analogy shows gender bias as it is associating a particular profession to the gender.

e) rich : young :: poor : elderly
In this case, the model is associating the age with the financial status of the individual. Thus, showing bias. 

"""


"""

2. What could be done to eliminate this bias from the embeddings? (200 word minimum, show word count.)

a) The bias in the word embeddings because of the training data can be removed if the bias is eliminated in the training data. The models should be trained on data which is free from gender stereotypes and other forms of stereotypes. 
b) Some of the cases show bias because of the constraint that the word cannot be the same. For example, in the case [ she : nurse :: he : physician ] the algorithm searches for the word that fits this analogy the best. It might not be because the model is associating a particular gender to the profession but because both nurse and physician are responsible providing care to the patients. Thus, the bias can be removed if we remove the constraint that it shouldn't be the same word.
c) The biases that we observe for the genders is generally because of the biased datasets. The word may be gender neutral but the algorithm may associate it with a particular gender because of the stereotype which is a result of the biased dataset as that particular word occurs more frequently in the context of one particular gender. If the word is made to appear more frequently in the context of the other gender, we can eliminate the bias.  

word count: 204

Sources - https://medium.com/linguaphile/on-gender-bias-in-word-embeddings-e53c40ba9294
	- https://cs229.stanford.edu/proj2016/report/BadieChakrabortyRudder-ReducingGenderBiasInWordEmbeddings-report.pdf

"""


"""

3. Should bias be eliminated from embeddings? Why or why not? (200 word minimum, show word count).


The biases should be eliminated to the most extent, especially if it is based on stereotypes of the society. The model should not associate a particular gender to certain occupations, or an individual's financial status to their emotional well being. As we have seen in word analogy examples, the model does that because of the bias and adheres to the stereotypes that exist in the society around us. Some words might be gender neutral, like certain occupations but they are associated to particular gender. In the word analogy: boy is to programmer as girl is to stylist shows prejudice between gender and occupation. The other examples in which the model show biases like associating the age of a person to their financial status, or the traditional beauty standards specific to a gender, or emotional well being of people to their financial status comes from how people in society associate words because of the bias. Some of these biases exist because people started to associate certain words with others over the years, thus creating the stereotypes that still exist. For example, in the word analogy: rich is to happy as poor is to tired shows the stereotype that the happiness of an individual is dependent on the wealth they possess. These are all biases that have existed in the society knowingly or unknowingly. 

word count: 222

"""
