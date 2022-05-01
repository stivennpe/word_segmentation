'''
Dataset creation
'''

from nltk import corpus
from nltk.corpus import brown
import nltk
nltk.download('brown')
nltk.download('punkt')
sents= brown.sents()
#!pip install wikipedia-api wikipedia
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)
wiki_1 = wiki_wiki.page("Opsin")
wiki_2 = wiki_wiki.page("Brown Bear")
wiki_3 = wiki_wiki.page("United Kingdom")
wiki_4 = wiki_wiki.page("Spain")
wiki_5 = wiki_wiki.page("PewDiePie")
wiki_6 = wiki_wiki.page("Civil conflict in the Philippines")
wiki_7 = wiki_wiki.page("Domenico Brescia")
wiki_8 = wiki_wiki.page("Qantara, Lebanon")
wiki_9 = wiki_wiki.page("Alright, Still")
wiki_10 = wiki_wiki.page("Chesterfield F.C.")
wiki_11 =wiki_wiki.page("France")
wiki_12 =wiki_wiki.page("United States")
texts= [wiki_1,wiki_2,wiki_3,wiki_4,wiki_5,wiki_6,wiki_7,wiki_8,wiki_9,wiki_10]

def clean(article): 
  data= article.text.split(".")
  for d in data:
    if d == "" or len(d)>0:
      data.remove(d)
    elif d== " ":
      data.remove(d)
    elif "\n" in d:
      d.replace("\n", "")
      
  return data

articles=[]

for text in texts:
  cleaned= clean(text)
  articles.append(cleaned)

final_articles=[]
for article in articles:
  for a in article:
    tokens= nltk.tokenize.word_tokenize(a)
    final_articles.append(tokens)

# Remove all empty lists from previous Wikipedia Data

for i,f in enumerate(final_articles):
  if f == "" or f == " ":
    del final_articles[i]
  elif len(f) <= 0:
    del final_articles[i]

#tagg in BIOES format

def conll_tagging(lista):
  tagged_list=[]
  for char in lista:
    temp=[]
    idx= len(char)
    for i,c in enumerate(char):
      if i ==0 and len(char)>1:
        temp.append([c, "B"])
      elif len(char) == 1:
        temp.append([c, "S"])
      elif i == idx-1:
        temp.append([c, "E"])
      elif len(char) == 1:
        temp.append([c, "S"])
      elif c.isspace() == True:
        temp.append([c, "X"])
      else:
        temp.append([c, "I"])
    tagged_list.append(temp)
  return tagged_list

finalf= sents+final_articles

#Calling the BIOES tagging function. Results should be as follows: [['T', 'h', 'e'], ['B', 'I', 'E']] 
total=[]
for sent in finalf:
  exa= conll_tagging(sent)
  sent=[]
  tempa=[]
  for f in exa:
    temp=[]
    for t in f:
      sent.append(t[0].lower())
      tempa.append(t[1])

  total.append([sent, tempa])

#Optional: If Character bigrams are to be used: e.g, [sc ch ho oo ol l- -l li if fe ex xp pe ec ct t...	B I I I I I I I I E B I I I I I I I E B E S]

from nltk.util import ngrams
twoChar=[]
for sent in finalf:
  temp=[]
  for s in sent:
    tempa=[]
    if len(s)>2:
      bigrams= ngrams(s,2)
      for item in bigrams:
        tempa.append("".join(item))
    else:
      tempa.append(s)
    temp.append(tempa)
  twoChar.append(temp)

tagged_list =[]
for sent in twoChar:
  temp=[]
  for word in sent:
    for i, bigram in enumerate(word):
      idx= len(word)
      if i ==0 and len(word)>1:
        temp.append([bigram, "B"])
      elif len(word) == 1:
        temp.append([bigram, "S"])
      elif i == idx-1:
        temp.append([bigram, "E"])
      elif len(bigram) == 1:
        temp.append([bigram, "S"])
      elif bigram.isspace() == True:
        temp.append([bigram, "X"])
      else:
        temp.append([bigram, "I"])
  tagged_list.append(temp)


total_bi=[]
for f in tagged_list:
  sent=[]
  tempa=[]
  for t in f:
    sent.append(t[0].lower())
    tempa.append(t[1])

  total_bi.append([sent, tempa])


# Create a .tsv file to store the dataset

from tqdm import tqdm
with open("brown_char.tsv", "a+") as f:

  for i in tqdm(range(len(total)),desc="Loading…",ascii=False, ncols=100):
    sent=" ".join(total[i][0])
    tag=" ".join(total[i][1])
    f.write(sent+"\t"+tag+"\n")

#Read
with open("brown_char.tsv", "r") as f:
  filea= f.readlines()
print(len(filea))


'''
Model Creation:
BiLSTM with CRF using pretrained embeddings.
'''
#!pip install pytorch torchtext==0.10.0
#!pip install pytorch-crf

import time
import torch
from torch import nn
from torch.optim import Adam
from torchtext.vocab import Vocab

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import spacy
from torchcrf import CRF
import numpy as np
import pandas as pd

import time
import random
import string
from itertools import chain

from torchtext.legacy import data
from tqdm import tqdm


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#No lower to use capitals as a feature
TEXT = data.Field(lower = False) #unigrams
TAG = data.Field(unk_token = None) 

fields = [('text', TEXT),('label', TAG)]

train_data = data.TabularDataset(
    path="/content/brown_char.tsv",
    format='tsv',
    fields=fields,
    skip_header= False,
)

#Test split
train_data, test_data = train_data.split(split_ratio=0.7, random_state = random.seed(SEED))
#Validation split
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

MIN_FREQ= 0 #To use all of the features
TEXT.build_vocab(train_data, 
                 min_freq = MIN_FREQ,
                 vectors = "glove.6B.100d", #pretrained embeddings 100 dimensions
                 unk_init = torch.Tensor.normal_)


TAG.build_vocab(train_data)

#Add "O" for visualization

TAG.vocab.itos.insert(5, "O")
TAG.vocab.stoi["O"]= 5

#to use the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device,
    sort=False
    )

# padding index
text_pad_id= TEXT.vocab.stoi[TEXT.pad_token]  
tag_pad_id= TAG.vocab.stoi[TAG.pad_token]

#Defining the model:

import torch.nn as nn
import torch.nn.functional as F
class BiLSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,word_pad_idx,tag_pad_idx):
        super().__init__()
        
        #Embedding layer chars
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(p=0.5)
        

        #BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True )
        
        #Linear Layer 
        self.fc_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        #CRF classifier
        self.tag_pad_idx = tag_pad_idx
        self.crf = CRF(num_tags=output_dim)
        
        # init poids avec distribution normale 
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, words, tags=None):

        # words = [sentence length, batch size]
        # tags = [sentence length, batch size]
        
        # embedding_out = [sentence length, batch size, embedding dim]
        embedding = self.emb_dropout(self.embedding(words))

        # lstm_out = [sentence length, batch size, hidden dim * 2]
        lstm_out, (hidden, cell) = self.lstm(embedding)
        
        # fc_out = [sentence length, batch size, output dim]
        fc_out = self.fc(self.fc_dropout(lstm_out))
        
        if tags is not None:
            mask = tags != self.tag_pad_idx
            crf_out = self.crf.decode(fc_out, mask=mask)

            crf_loss = -self.crf(fc_out, tags=tags, mask=mask) 
        else:
            crf_out = self.crf.decode(fc_out)
            crf_loss = None
          
        return crf_out , crf_loss

#Init model 

embedding_dim=100
tag_pad_idx=tag_pad_id
model = BiLSTM(
    input_dim=len(TEXT.vocab),
    embedding_dim=100,
    hidden_dim=256,
    output_dim=len(TAG.vocab),
    word_pad_idx=text_pad_id,
    tag_pad_idx=tag_pad_id
)
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        
model.apply(init_weights)
#Use of glove pretrained embeddings
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[tag_pad_idx] = torch.zeros(embedding_dim)

#Optimizer custom function

def optimizer_func(model, lr=1e-5, eps=1e-6, weight_decay_rate=0.001, second_weight_decay_rate=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': second_weight_decay_rate}]
    return optim.Adam(
        optimizer_grouped_parameters,
        lr=lr,
        eps=eps
    )

optimizer = optimizer_func(model, lr=1e-5, eps=1e-6, weight_decay_rate=0.001, second_weight_decay_rate=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss(ignore_index = tag_pad_id)
model = model.to(device)

## Metrics

from sklearn.metrics import f1_score, classification_report

def f1_loss( preds, y, tag_pad_idx, full_report=False):
    index_o = TAG.vocab.stoi["O"]
    positive_labels = [i for i in range(len(TAG.vocab.itos))
                       if i not in (tag_pad_idx, index_o)]

    flatten_preds = [pred for sent_pred in preds for pred in sent_pred]

    positive_preds = [pred for pred in flatten_preds
                      if pred not in (tag_pad_idx, index_o)]

    flatten_y = [tag for sent_tag in y for tag in sent_tag]
    if full_report:
      
        positive_names = [TAG.vocab.itos[i]
                              for i in range(len(TAG.vocab.itos))
                              if i not in (tag_pad_idx, index_o)]
        print(classification_report(
                y_true=flatten_y,
                y_pred=flatten_preds,
                labels=positive_labels,
                target_names=positive_names
            ))

    return f1_score(
            y_true=flatten_y,
            y_pred=flatten_preds,
            labels=positive_labels,
            average="micro"
        ), flatten_preds, flatten_y


#Training Function

def train(model, iterator, optimizer, tag_pad_idx):
    
    epoch_loss = 0
    epoch_f1 = 0    
    model.train()
    
    for batch in iterator:
        
        text = batch.text
        tags = batch.label
        optimizer.zero_grad()

        pred_tags_list, batch_loss = model(text, tags)
        
        # pour calculer la loss et le score f1, on flatten true tags
        true_tags_list = [
                [tag for tag in sent_tag if tag != tag_pad_idx]
                for sent_tag in tags.permute(1, 0).tolist()
            ]
        f1,_,_ = f1_loss(pred_tags_list, true_tags_list, tag_pad_idx)

        batch_loss.backward()
        
        optimizer.step()
        epoch_loss += batch_loss.item()
        epoch_f1 += f1
        
    return epoch_loss / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, tag_pad_idx,full_report):
    
    epoch_loss = 0
    epoch_f1 = 0
    
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
    
        for batch in iterator:

            text = batch.text
            tags = batch.label

            pred_tags_list, batch_loss = model(text, tags)
            true_tags_list = [
                [tag for tag in sent_tag if tag != tag_pad_id]
                for sent_tag in tags.permute(1, 0).tolist()
                ]
            
            f1, pred, lab = f1_loss(pred_tags_list, true_tags_list, tag_pad_idx, full_report)
            preds.append(pred)
            labels.append(lab)
            epoch_loss += batch_loss.item()
            epoch_f1 += f1
        
    return epoch_loss / len(iterator), epoch_f1 / len(iterator),preds, labels


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 20

t_loss = []
t_f1 = []
v_loss = []
v_f1 = []

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
   
    
    train_loss, train_f1 = train(model, train_iterator, optimizer, tag_pad_id)
    t_loss.append(train_loss)
    t_f1.append(train_f1) 
    
    valid_loss, valid_f1,_,_ = evaluate(model, valid_iterator, tag_pad_id, full_report= False)
    v_loss.append(valid_loss)
    v_f1.append(valid_f1)
    
    scheduler.step()
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pt')
    

    if epoch%1 == 0: 
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train F1 score: {train_f1*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1 score: {valid_f1*100:.2f}%')


#Loss visualization
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
sns.set()

x = np.linspace(0, N_EPOCHS,N_EPOCHS)

plt.plot(x,t_loss)
plt.plot(x,v_loss)
plt.title("Loss")
plt.legend(["Train loss", "Valid loss"])

#f1 Visualization
x = np.linspace(0, N_EPOCHS,N_EPOCHS)

plt.plot(x,t_f1)
plt.plot(x,v_f1)
plt.title("F1 score")
plt.legend(["Train F1", "Valid F1"])


#Evaluate the model on the test split

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
model.load_state_dict(torch.load('best_model.pt'))

test_loss, test_f1, preds, labels = evaluate(model, test_iterator, tag_pad_id, full_report=False)
print(f'Test Loss: {test_loss:.3f} |  Test F1 score: {test_f1*100:.2f}%')

predict =  [item for sublist in preds for item in sublist]
true =  [item for sublist in labels for item in sublist]
confusion = confusion_matrix(true, predict)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print('Classification Report:')
print(classification_report(true, predict, labels=[4, 3, 2, 1], digits=4))
    
cm = confusion_matrix(true, predict, labels=[4, 3, 2, 1])
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

#Save model's vocab for future inference

def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()

save_vocab(TEXT.vocab,"/content/vocab.pkl")
save_vocab(TAG.vocab,"/content/vocab_tag.pkl")

'''
Inference:
To make new predictions with unseen data using the trained model.
'''

import pickle
vocab_text_p= "/content/vocab.pkl"
vocab_tag_p= "/content/vocab_tag.pkl"
vocabulary_text=pickle.load(open(vocab_text_p, "rb"))
vocabulary_tag=pickle.load(open(vocab_tag_p, "rb"))


#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# padding index
text_pad_id= vocabulary_text.stoi["<pad>"]  
tag_pad_id= vocabulary_tag.stoi["<pad>"]
#model instance
model = BiLSTM(
    input_dim=len(vocabulary_text),
    embedding_dim=100,
    hidden_dim=256,
    output_dim=len(vocabulary_tag),
    word_pad_idx=text_pad_id,
    tag_pad_idx=tag_pad_id
).to(device)
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('best_model.pt'))

from nltk.util import ngrams

#Custom tokenizer for uni or bi characther n-grams

def char_tok_pred(words):
  uniChar=[]
  twoChar=[]
  for word in words:
    if len(word)>0:

      temporal=[]
      #uni
      unigrams = ngrams(word, 1)
      for item in unigrams:
          temporal.append("".join(item))
      uniChar+=temporal
      #bi
  temporal=[]
  bigrams= ngrams(words,2)
  if len(words)>2:
    for item in bigrams:
        temporal.append("".join(item))
    twoChar+=temporal

  return uniChar, twoChar

#Character uni grams returned the best results. Leaving unigrams as default but bigrams can be fed too.

def predict(model, device, sentence, text_field, tag_field):
    
    model.eval()
  

    tokens_uni, tokens_two= char_tok_pred(sentence)
    #print(tokens_uni)
    max_word_len = max([len(token) for token in tokens_uni])

    numericalized_tokens = [text_field.stoi[t] for t in tokens_uni]
    unk_idx = text_field.stoi["<unk>"]  
    unks = [t for t, n in zip(tokens_uni, numericalized_tokens) if n == unk_idx]
    
    token_tensor = torch.as_tensor(numericalized_tokens)    
    token_tensor = token_tensor.unsqueeze(-1).to(device)

    predictions, _ = model(token_tensor)
    predicted_tags = [vocabulary_tag.itos[t] for t in predictions[0]]
    
    return tokens_uni, predicted_tags, unks

#sentence=["ThisfilmisTerrible."]

tokens, pred_tags, unks = predict(model, 
                                       device, 
                                       sentence, 
                                       vocabulary_text, 
                                       vocabulary_tag
                                      )
print(pred_tags)

#['T', 'h', 'i', 's', 'f', 'i', 'l', 'm', 'i', 's', 't', 'e', 'r', 'r', 'i', 'b', 'l', 'e'] Sentence chars
#['B', 'I', 'I', 'E', 'B', 'I', 'I', 'I', 'B', 'E', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'E'] Predicted tags
#['B', 'I', 'I', 'E', 'B', 'I', 'I', 'E', 'B', 'E', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'E'] Real tags

#reconstruct the sentence:
def reconstruct(tokens, pred_tags):
    final= []
    for i,t in zip(tokens, pred_tags):
        if t == "E":
            final.append(i)
            final.append(" ")
        elif t== "S":
            if i in ".,?!¿¡:;()[]{}'$/-_%":
                final.append(i)
                final.append(" ")
            else:
                final.append(" ")
                final.append(i)
                final.append(" ")
        elif t == "B":
            final.append(i)
        else:
            final.append(i)
    return "".join(final)

sentence= reconstruct(tokens, pred_tags)
