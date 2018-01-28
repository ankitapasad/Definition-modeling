#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Steps:
# 1. Read all the word-embedding pairs in a list and a numpy matrix
# 2. Read the vocab into a list
# 3. Read the word-definition pairs into the 2-d list

# Training process:
# 1. start timer
# 2. initialize criterion and optimizers
# 3. create set of training pairs
# 4. start empty losses array for plotting

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

SOS_token = 0
EOS_token = 1
use_cuda = torch.cuda.is_available()

## 1. Read all the word-embedding pairs in a list and a numpy matrix
print("Reading word embedding data\n")
sys.stdout.flush()
emb = open('defModeling/pre-trainedEmbeddings/glove.840B.19L.300d.txt').readlines()
emb_words = []
emb_vectors = np.zeros([len(emb),300])
emb_words.append(emb[i].split(' ')[0] for i in (1,len(emb)))
for i in range (0,len(emb)):
    emb_list = emb[i].split(' ')
    emb_words.append(emb_list[0])
    emb_vectors[i] = map(float,emb_list[1:]) # 300-dim vector

# np.savetxt('../data/readableFormat/emb_vectors.txt',emb_vectors)
# np.savetxt('../data/readableFormat/emb_words.txt',emb_words,fmt="%s")

## 2. Read the vocab into a list
print("Reading vocabulary")
# sys.stdout.flush()
vocab = open('defModeling/data/commondefs/vocab.txt').readlines()
for i in range(0,len(vocab)):
    vocab[i] = vocab[i].strip('\n')

## 3. Read the word-definition pairs into the 2-d list
def readDefs(fileName):
    data = open('defModeling/data/commondefs/'+fileName+'.txt').readlines()
    words = []
    definitions = [] # list of lists
    for i in range(len(data)):
        words.append(data[i].split('\t')[0])
        definition = ((data[i].split('\t')[3]).strip('\n')).split(' ')
        definitions.append(definition)

    return words, definitions

# print("Loading word embedding data\n")
# emb_vectors = np.loadtxt('../data/readableFormat/emb_vectors.txt')
# emb_words = np.loadtxt('../data/readableFormat/emb_words.txt',dtype='str',encoding='utf-8')

print("Reading word-def pairs\n")
# sys.stdout.flush()
[trainWords,trainDefs] = readDefs('train')
[valWords,valDefs] = readDefs('valid')
[testWords,testDefs] = readDefs('test')

MAX_LENGTH = 0
for i in range(len(trainDefs)):
    if(len(trainDefs[i])>MAX_LENGTH):
        MAX_LENGTH = len(trainDefs[i])

## Change all definitions into a list of indices
def def2index(definitions):
    defIndex = []
    for i in range(len(definitions)):
        thisDefIndex = []
        for j in range(len(definitions[i])):
            if(definitions[i][j] in vocab):
                thisDefIndex.append(vocab.index(definitions[i][j]))
            else:
                # vocab.append(definitions[i][j])
                thisDefIndex.append(len(vocab)) # OOV words
                print("new word encountered: %s", definitions[i][j])
        thisDefIndex.append(EOS_token)
        defIndex.append(thisDefIndex)

    return defIndex

print("Transforming natural language definitions to indices")
# sys.stdout.flush()
trainDefIndex = def2index(trainDefs)
valDefIndex = def2index(valDefs)
testDefIndex = def2index(testDefs)

## Embedding2seq model

# The initial input token is the start-of-string <SOS> token, and the first 
# initial hidden state of decoder is the context vector (the encoder’s last output vector).
# each unit of a decoder is given an input token and a hidden state

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.affine = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, embedding):
        output = self.embedding(input).view(1, 1, -1)
        # output = torch.cat((output[0], embedding[0]), 1) # embedding as an input to each layer
        output = self.affine(output)
        for i in range(self.n_layers):
            # print(i, output)
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# Training with teacher forcing - for fast convergence

teacher_forcing_ratio = 0.5


def train(decoder_hidden, embedding, target_variable, decoder, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # encoder_hidden = encoder.initHidden()

    # encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    # encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(
    #         input_variable[ei], encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # decoder_hidden = encoder_hidden # last encoder hidden state

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            print("in decoder")
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, embedding)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            print("in decoder")
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, embedding)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()
    print("Loss = ", loss)

    # encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(decoder, n_epoch , print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_every_val = 1000
    n_iters = len(trainWords)
    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # for epoch in range (n_epoch):
        # print("epoch_no = ", epoch)
        # order = np.random.permutation(len(trainWords))
## On train data
    for iter in range(1, len(trainWords) + 1):
        print(iter)
        training_word = random.choice(trainWords)
        training_defIndex = trainDefIndex[trainWords.index(training_word)]
        # training_word = trainWords[order[iter-1]]
        # training_defIndex = trainDefIndex[order[iter-1]]
        if(training_word in emb_words):
            input_embedding = emb_vectors[emb_words.index(training_word)] # 300-dim list
        else:
            print("embedding not present for word \t", training_word,"\n")
            input_embedding = (np.random.uniform(low=-1.5,high=2.6,size=(300,))).tolist()
        hidden_variable = (np.random.uniform(low=-1.5,high=2.6,size=(300,))).tolist()
        embedding = Variable(torch.FloatTensor(input_embedding).view(1,1,-1))
        hidden_variable = Variable(torch.FloatTensor(hidden_variable).view(1,1,-1))
        target_variable = Variable(torch.LongTensor(training_defIndex).view(-1,1))
        embedding = embedding.cuda() if use_cuda else embedding
        hidden_variable = hidden_variable.cuda() if use_cuda else hidden_variable
        target_variable = target_variable.cuda() if use_cuda else target_variable
        print("in train")
        loss = train(hidden_variable, embedding, target_variable, decoder, decoder_optimizer, criterion)
        print("training done")
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # print('%s' %(timeSince(start, epoch / n_epoch)))

    showPlot(plot_losses)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('Loss_dm.png')

def evaluate(decoder, embedding, decoder_hidden, max_length=MAX_LENGTH):

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, embedding)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(vocab[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words


def evaluateRandomly(decoder, n=10):
    for i in range(n):
        print("test iter: ", i)
        test_word = random.choice(testWords)
        target_def = testDefs[testWords.index(test_word)]
        target_sentence = ' '.join(target_def)
        print("Word: ", test_word)
        print("Actual def: ", target_sentence)
        if(test_word in emb_words):
            input_embedding = emb_vectors[emb_words.index(test_word)] # 300-dim list
        else:
            print("embedding not present for word \t", test_word)
            input_embedding = np.random.uniform(low=-1.5,high=2.6,size=(300,)).tolist()

        hidden_variable = (np.random.uniform(low=-1.5,high=2.6,size=(300,))).tolist()
        hidden_variable = Variable(torch.FloatTensor(hidden_variable).view(1,1,-1))
        embedding = Variable(torch.FloatTensor(input_embedding).view(1,1,-1))
        embedding = embedding.cuda() if use_cuda else embedding
        hidden_variable = hidden_variable.cuda() if use_cuda else hidden_variable
        output_words = evaluate(decoder, embedding, hidden_variable)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence, "\n")

def evaluateOnTrainData(decoder, n=10):
    for i in range(n):
        print("test_on_train iter: ", i)
        test_word = random.choice(trainWords)
        target_def = trainDefs[trainWords.index(test_word)]
        target_sentence = ' '.join(target_def)
        print("Word: ", test_word)
        print("Actual def: ", target_sentence)
        if(test_word in emb_words):
            input_embedding = emb_vectors[emb_words.index(test_word)] # 300-dim list
        else:
            print("embedding not present for word \t", test_word)
            input_embedding = np.random.uniform(low=-1.5,high=2.6,size=(300,)).tolist()
        
        hidden_variable = (np.random.uniform(low=-1.5,high=2.6,size=(300,))).tolist()
        hidden_variable = Variable(torch.FloatTensor(hidden_variable).view(1,1,-1))
        embedding = Variable(torch.FloatTensor(input_embedding).view(1,1,-1))
        embedding = embedding.cuda() if use_cuda else embedding
        hidden_variable = hidden_variable.cuda() if use_cuda else hidden_variable
        output_words = evaluate(decoder, embedding, hidden_variable)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence, "\n")

print("TRAINING")
hidden_size = 300
# encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               # 1, dropout_p=0.1)
decoder1 = DecoderRNN(hidden_size,len(vocab),1)
n_epoch = 5

if use_cuda:
#     encoder1 = encoder1.cuda()
#     attn_decoder1 = attn_decoder1.cuda()
    decoder1 = decoder1.cuda()

trainIters(decoder1, n_epoch, print_every=5000)

print("TESTING")
evaluateOnTrainData(decoder1)
evaluateRandomly(decoder1)

sys.stdout.flush()
plt.show()



