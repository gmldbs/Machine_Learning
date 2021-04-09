import json
import numpy as np
import re
import math
import random
from matplotlib import pyplot as plt
# pickle is used to save 3000 frequently list data
#import pickle
#import os

Funny_words = {}
Useful_words = {}
Cool_words = {}
Positive_words = {}
Not_Funny_words = {}
Not_Useful_words = {}
Not_Cool_words = {}
Not_Positive_words = {}
num_Funny = 0
num_Useful = 0
num_Cool = 0
num_Positive = 0
num_Not_Funny = 0
num_Not_Useful = 0
num_Not_Cool = 0
num_Not_Positive = 0
total_Funny = 0
total_Not_Funny = 0
total_Useful = 0
total_Not_Useful = 0
total_Cool = 0
total_Not_Cool = 0
total_Positive = 0
total_Not_Positive = 0
word_list = []
# load reviews that total votes is between 3 and 10
def load(path):
    review = []
    for line in open(path,'r'):
        row = json.loads(line)
        total_vote = row['votes']['funny'] + row['votes']['useful'] + row['votes']['cool']
        if total_vote >= 3  and total_vote <=10:
            review.append(row)
    review = random.sample(review, 16000)
    return review
# preprocessing each sentence in review text
def preprocessing(sentence):
    letters = re.sub('[^a-zA-Z]',' ',sentence)
    words_list = letters.lower().split()
    return words_list

# this part is learning by input reviews.
def dataset_learn(reviews):
    global total_Not_Funny, total_Not_Useful, total_Not_Cool, total_Not_Positive, total_Funny, total_Useful, total_Cool, total_Positive, num_Funny, num_Useful, num_Cool, num_Positive, Funny_words, Useful_words, Cool_words, Positive_words, Not_Funny_words, Not_Useful_words, Not_Cool_words, Not_Positive_words, word_list
    global num_Not_Funny, num_Not_Useful, num_Not_Cool, num_Not_Positive

    #initialize part
    Funny_words = {}
    Useful_words = {}
    Cool_words = {}
    Positive_words = {}
    Not_Funny_words = {}
    Not_Useful_words = {}
    Not_Cool_words = {}
    Not_Positive_words = {}
    num_Funny = 0
    num_Useful = 0
    num_Cool = 0
    num_Positive = 0
    num_Not_Funny = 0
    num_Not_Useful = 0
    num_Not_Cool = 0
    num_Not_Positive = 0
    total_Cool = 0
    total_Funny = 0
    total_Positive = 0
    total_Useful = 0
    total_Not_Cool = 0
    total_Not_Funny = 0
    total_Not_Positive = 0
    total_Not_Useful = 0

    #Laplace Smoothing
    for word in word_list:
        Funny_words[word] = 1
        Useful_words[word] = 1
        Cool_words[word] = 1
        Positive_words[word] = 1
        Not_Funny_words[word] = 1
        Not_Useful_words[word] = 1
        Not_Cool_words[word] = 1
        Not_Positive_words[word] = 1
    
    for review in reviews:
        review = dict(review)
        isFunny = 0
        isUseful = 0
        isCool = 0
        isPositive = 0
        # check this review's type
        if review["votes"]["funny"] > 0: 
            isFunny = 1
            num_Funny += 1
        else:
            num_Not_Funny += 1
        if review["votes"]["useful"] > 0: 
            isUseful = 1
            num_Useful += 1
        else:
            num_Not_Useful += 1
        if review["votes"]["cool"] > 0: 
            isCool = 1
            num_Cool += 1
        else:
            num_Not_Cool += 1
        if review["stars"] > 3.5: 
            isPositive = 1
            num_Positive += 1
        else:
            num_Not_Positive += 1
        text = review["text"]
        words_in_text = preprocessing(text)

        # make the set data type to binary naive bayes
        words_in_text = set(words_in_text)

        # learning
        for word in words_in_text:
            if word not in word_list:
                continue
            if isFunny:
                Funny_words[word] += 1
            else:
                Not_Funny_words[word] += 1
            
            if isUseful:
                Useful_words[word] += 1
            else:
                Not_Useful_words[word] += 1

            if isCool:
                Cool_words[word] += 1
            else:
                Not_Cool_words[word] += 1
            
            if isPositive :
                Positive_words[word] += 1
            else:
                Not_Positive_words[word] += 1
    
    # Check the word that is included label, not_label is exist or not.
    # If the word that isn't included label and not_label, the word is removed in the word list in that label.
    # ex) If the 'test' isn't exist Funny and not Funny, 'test' is removed in Funny and Not_Funny word list.
    for word in word_list:
        if Funny_words[word] == 1 and Not_Funny_words[word] == 1:
            del Funny_words[word]
            del Not_Funny_words[word]

        if Useful_words[word] == 1 and Not_Useful_words[word] == 1:
            del Useful_words[word]
            del Not_Useful_words[word]

        if Cool_words[word] == 1 and Not_Cool_words[word] == 1:
            del Cool_words[word]
            del Not_Cool_words[word]

        if Positive_words[word] == 1 and Not_Positive_words[word] == 1:
            del Positive_words[word]
            del Not_Positive_words[word]
    
    # Calculate Total number of words in each label.
    for word in Funny_words:
        total_Funny += Funny_words[word]
    for word in Not_Funny_words:
        total_Not_Funny += Not_Funny_words[word]

    for word in Useful_words:
        total_Useful += Useful_words[word]
    for word in Not_Useful_words:
        total_Not_Useful += Not_Useful_words[word]

    for word in Cool_words:
        total_Cool += Cool_words[word]
    for word in Not_Cool_words:
        total_Not_Cool += Not_Cool_words[word]

    for word in Positive_words:
        total_Positive += Positive_words[word]
    for word in Not_Positive_words:
        total_Not_Positive += Not_Positive_words[word]

# this function is used to make dataset that is frequently 3000 words
def make_dataset(path):
    global word_list
    reviews = load(path)
    word_dic = {}
    for review in reviews:
        text = review["text"]
        words_in_text = preprocessing(text)
        for word in words_in_text:
            if word in word_dic:
                word_dic[word] += 1
            else:
                word_dic[word] = 1
    word_list = sorted(word_dic.items(), key=lambda  x: x[1], reverse = True)
    word_list = dict(word_list[:3000])
# this is the testing part.
# input is test reviews and test review length
def testing(reviews, length):
    global total_Not_Funny, total_Not_Useful, total_Not_Cool, total_Not_Positive, total_Funny, total_Useful, total_Cool, total_Positive, num_Funny, num_Useful, num_Cool, num_Positive, Funny_words, Useful_words, Cool_words, Positive_words, Not_Funny_words, Not_Useful_words, Not_Cool_words, Not_Positive_words, word_list
    global loss_Funny, loss_Useful, loss_Cool, loss_Positive
    global num_Not_Funny, num_Not_Useful, num_Not_Cool, num_Not_Positive
    correct_Funny = 0
    correct_Useful = 0
    correct_Cool = 0
    correct_Positive = 0
    for review in reviews:
        review = dict(review)
        text = review["text"]
        isFunny = 0
        isUseful = 0
        isCool = 0
        isPositive = 0
        p_Funny = 0
        p_Not_Funny = 0
        p_Useful = 0
        p_Not_Useful = 0
        p_Cool = 0
        p_Not_Cool = 0
        p_Positive = 0
        p_Not_Positive = 0
        # check the number of word list in each label 
        if num_Funny == 0: 
            p_Funny = -1000000
        else: 
            p_Funny = math.log(num_Funny/length)
        if num_Not_Funny == 0:
            p_Not_Funny = -1000000
        else: 
            p_Not_Funny = math.log(num_Not_Funny/length)

        if num_Useful == 0:
            p_Useful = -1000000
        else : 
            p_Useful = math.log(num_Useful/length)
        if num_Not_Useful == 0:
            p_Not_Useful = -1000000
        else: 
            p_Not_Useful = math.log(num_Not_Useful/length)

        if num_Cool == 0:
            p_Cool = -1000000
        else:
            p_Cool = math.log(num_Cool/length)

        if num_Not_Cool == 0:
            p_Not_Cool = -1000000
        else: 
            p_Not_Cool = math.log(num_Not_Cool/length)

        if num_Positive == 0:
            p_Positive = -1000000
        else:
            p_Positive = math.log(num_Positive/length)

        if num_Not_Positive == 0:
            p_Not_Positive = -1000000
        else:
            p_Not_Positive = math.log(num_Not_Positive/length)

        if review["votes"]["funny"] > 0: 
            isFunny = 1
        if review["votes"]["useful"] > 0: 
            isUseful = 1
        if review["votes"]["cool"] > 0: 
            isCool = 1
        if review["stars"] > 3.5: 
            isPositive = 1
        words_in_text = preprocessing(text)
        words_in_text = set(words_in_text)
        # Calculate the prob label and not label
        for word in words_in_text:
            if word in Funny_words:
                p_Funny += math.log(Funny_words[word]/total_Funny)
                p_Not_Funny += math.log(Not_Funny_words[word]/total_Not_Funny)
            if word in Useful_words:
                p_Useful += math.log(Useful_words[word]/total_Useful)
                p_Not_Useful += math.log(Not_Useful_words[word]/total_Not_Useful)
            if word in Cool_words:
                p_Cool += math.log(Cool_words[word]/total_Cool)
                p_Not_Cool += math.log(Not_Cool_words[word]/total_Not_Cool)
            if word in Positive_words:
                p_Positive += math.log(Positive_words[word]/total_Positive)
                p_Not_Positive += math.log(Not_Positive_words[word]/total_Not_Positive)
        # check the predict is correct or not
        if(p_Funny>p_Not_Funny):
            if(isFunny == 1):
                correct_Funny += 1
        else:
            if(isFunny == 0):
                correct_Funny += 1

        if(p_Useful>p_Not_Useful):
            if(isUseful == 1):
                correct_Useful += 1
        else:
            if(isUseful == 0):
                correct_Useful += 1

        if(p_Cool>p_Not_Cool):
            if(isCool == 1):
                correct_Cool += 1
        else:
            if(isCool == 0):
                correct_Cool += 1

        if(p_Positive>p_Not_Positive):
            if(isPositive == 1):
                correct_Positive += 1
        else:
            if(isPositive == 0):
                correct_Positive += 1
    # print the accuracy of this test cases
    print("Number of test case : 3200")
    print("Number of correct Funny case : {}".format(correct_Funny))
    print("Funny case accuarcy  = {}\n".format(correct_Funny/3200))
    print("Number of correct Useful case : {}".format(correct_Useful))
    print("Useful case accuarcy  = {}\n".format(correct_Useful/3200))
    print("Number of correct Cool case : {}".format(correct_Cool))
    print("Cool case accuarcy  = {}\n".format(correct_Cool/3200))
    print("Number of correct Positive case : {}".format(correct_Positive))
    print("Positive case accuarcy  = {}\n".format(correct_Positive/3200))
    # append zero-one loss of this result to print plot graph
    loss_Funny.append((3200-correct_Funny)/3200)
    loss_Useful.append((3200-correct_Useful)/3200)
    loss_Cool.append((3200-correct_Cool)/3200)
    loss_Positive.append((3200-correct_Positive)/3200)

# this function is used to default predict 
def testing_default(reviews, length):
    global total_Not_Funny, total_Not_Useful, total_Not_Cool, total_Not_Positive, total_Funny, total_Useful, total_Cool, total_Positive, num_Funny, num_Useful, num_Cool, num_Positive, Funny_words, Useful_words, Cool_words, Positive_words, Not_Funny_words, Not_Useful_words, Not_Cool_words, Not_Positive_words, word_list
    global loss_Funny, loss_Useful, loss_Cool, loss_Positive
    global num_Not_Funny, num_Not_Useful, num_Not_Cool, num_Not_Positive
    correct_Funny = 0
    correct_Useful = 0
    correct_Cool = 0
    correct_Positive = 0
    for review in reviews:
        review = dict(review)
        text = review["text"]
        isFunny = 0
        isUseful = 0
        isCool = 0
        isPositive = 0
        if review["votes"]["funny"] > 0: 
            isFunny = 1
        if review["votes"]["useful"] > 0: 
            isUseful = 1
        if review["votes"]["cool"] > 0: 
            isCool = 1
        if review["stars"] > 3.5: 
            isPositive = 1
        words_in_text = preprocessing(text)
        words_in_text = set(words_in_text)
        if(num_Funny > num_Not_Funny):
            if(isFunny == 1):
                correct_Funny += 1
        else:
            if(isFunny == 0):
                correct_Funny += 1

        if(num_Useful > num_Not_Useful):
            if(isUseful == 1):
                correct_Useful += 1
        else:
            if(isUseful == 0):
                correct_Useful += 1

        if(num_Cool > num_Not_Cool):
            if(isCool == 1):
                correct_Cool += 1
        else:
            if(isCool == 0):
                correct_Cool += 1

        if(num_Positive>num_Not_Positive):
            if(isPositive == 1):
                correct_Positive += 1
        else:
            if(isPositive == 0):
                correct_Positive += 1
    # print the accuracy of this test cases
    print("\nDefault Error Case")
    print("Number of test case : 3200")
    print("Number of correct Funny case : {}".format(correct_Funny))
    print("Funny case accuarcy  = {}\n".format(correct_Funny/3200))
    print("Number of correct Useful case : {}".format(correct_Useful))
    print("Useful case accuarcy  = {}\n".format(correct_Useful/3200))
    print("Number of correct Cool case : {}".format(correct_Cool))
    print("Cool case accuarcy  = {}\n".format(correct_Cool/3200))
    print("Number of correct Positive case : {}".format(correct_Positive))
    print("Positive case accuarcy  = {}\n".format(correct_Positive/3200))
    # append zero-one loss of this result to print plot graph
    default_loss_Funny.append((3200-correct_Funny)/3200)
    default_loss_Useful.append((3200-correct_Useful)/3200)
    default_loss_Cool.append((3200-correct_Cool)/3200)
    default_loss_Positive.append((3200-correct_Positive)/3200)

loss_Funny = []
loss_Useful = []
loss_Cool = []
loss_Positive = []
default_loss_Funny = []
default_loss_Useful = []
default_loss_Cool = []
default_loss_Positive = []
test_case = [50,100,200,400,800,1600,3200,6400,12800]
#make frequency 3000 words list
print("making dataset frequently 3000 words...")
make_dataset('yelp_academic_dataset_review.json')
print("making dataset process end!")
'''
if os.path.isfile('list.txt'):
    print("already prepare dataset frequently 3000 words list")
    print("loading dataset frequently 3000 words...")
    with open("list.txt",'rb') as f:
        word_list = pickle.load(f)
    print("loading dataset frequently 3000 words process end!")
    #print("\nmaking training and test dataset...")
else:
    #make frequency 3000 words list
    print("making dataset frequently 3000 words...")
    make_dataset('yelp_academic_dataset_review.json')
    print("save list of frequently 3000 words data...")
    #And then save the list.txt to save time when after running
    with open('list.txt', 'wb') as f:
        pickle.dump(word_list, f)
    print("making dataset process end!")
'''
#loading saving 3000 words
'''
print("loading dataset frequently 3000 words...")
with open("list.txt",'rb') as f:
    word_list = pickle.load(f)
print("loading dataset frequently 3000 words process end!")
print("\nmaking training and test dataset...")
'''
reviews = load('yelp_academic_dataset_review.json')

print("making trainging and test dataset process end!")
#learn each training case
for cur_test in test_case:
    print ("test case size : {}".format(cur_test))
    dataset_learn(reviews[3200:3200+cur_test])
    testing(reviews[:3200],cur_test)
    testing_default(reviews[:3200],cur_test)

#sample = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#print the graph 
sample = [50,100,200,400,800,1600,3200,6400,12800]
print('loss graph (max dataset size 12800)')
plt.plot(sample, loss_Funny)
plt.plot(sample, loss_Useful)
plt.plot(sample, loss_Cool)
plt.plot(sample, loss_Positive)
plt.xlabel('Label')
plt.ylabel('Loss')
plt.title('Loss Graph (max dataset size 12800)')
plt.legend(['Funny', 'Useful','Cool','Positive'])
plt.show()
#print default error
plt.plot(sample, default_loss_Funny)
plt.plot(sample, default_loss_Useful)
plt.plot(sample, default_loss_Cool)
plt.plot(sample, default_loss_Positive)
plt.xlabel('Label')
plt.ylabel('Loss')
plt.title('Default Loss Graph (max dataset size 12800)')
plt.legend(['Funny', 'Useful','Cool','Positive'])
plt.show()
