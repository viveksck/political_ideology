import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import math
def to_var(x):                                                                                                                                                                   
    if torch.cuda.is_available():                                                                                                                                                
        x = x.cuda()                                                                                                                                                             
    return Variable(x)                                                                                                                                                           


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


class AttentionWordRNN(nn.Module):
    
    
    def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden, bidirectional= True):        
        
        super(AttentionWordRNN, self).__init__()
        
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        
        self.lookup = nn.Embedding(num_tokens, embed_size)
        if bidirectional == True:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2* word_gru_hidden,2*word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2* word_gru_hidden,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2*word_gru_hidden, 1))
        else:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
            
        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)

        
        
    def forward(self, embed, state_word):
        # embeddings
        embedded = self.lookup(embed)
        # word level gru
        output_word, state_word = self.word_gru(embedded, state_word)
        # print output_word.size()
        word_squish = batch_matmul_bias(output_word, self.weight_W_word,self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1,0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1,0))        
        return word_attn_vectors, state_word, word_attn_norm


    def init_hidden(self, batch_size=None):
        if batch_size is None:
            if self.bidirectional == True:
                return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
            else:
                return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))        
        else:
            if self.bidirectional == True:
                return Variable(torch.zeros(2, batch_size, self.word_gru_hidden))
            else:
                return Variable(torch.zeros(1, batch_size, self.word_gru_hidden))        


# ## Sentence Attention model with bias

class AttentionSentRNN(nn.Module):
    
    
    def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional=True, return_softmax=False):        
        
        super(AttentionSentRNN, self).__init__()
        
        self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.return_softmax=return_softmax
        
        if bidirectional == True:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional= True)        
            self.weight_W_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden ,2* sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden,1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden, 1))
            self.final_linear = nn.Linear(2* sent_gru_hidden, n_classes)
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional= False)        
            self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden ,sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden,1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.final_linear = nn.Linear(sent_gru_hidden, n_classes)
        self.softmax_sent = nn.Softmax()
        self.final_softmax = nn.Softmax()
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1,0.1)
        
        
    def forward(self, word_attention_vectors, state_sent):
        output_sent, state_sent = self.sent_gru(word_attention_vectors, state_sent)        
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent,self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn.transpose(1,0))
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1,0))
        final_map = self.final_linear(sent_attn_vectors.squeeze(0))
        if self.return_softmax:
            return F.softmax(final_map), state_sent, sent_attn_norm
        else:
            return final_map, state_sent, sent_attn_norm

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            if self.bidirectional == True:
                return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
            else:
                return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden))        
        else:
            if self.bidirectional == True:
                return Variable(torch.zeros(2, batch_size, self.sent_gru_hidden))
            else:
                return Variable(torch.zeros(1, batch_size, self.sent_gru_hidden))        

    
class AttentionSentRNNVAE(nn.Module):
    def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional=True, return_softmax=False):        
        
        super(AttentionSentRNNVAE, self).__init__()
        
        self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.return_softmax=return_softmax
        
        if bidirectional == True:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, 2*sent_gru_hidden, bidirectional= True)        
            self.weight_W_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden ,2* sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden,1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden, 1))
            self.final_linear = nn.Linear(2* sent_gru_hidden, n_classes)
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, 2*sent_gru_hidden, bidirectional= True)        
            self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden ,sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden,1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.final_linear = nn.Linear(sent_gru_hidden, n_classes)
        self.softmax_sent = nn.Softmax()
        self.final_softmax = nn.Softmax()
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1,0.1)

    def reparameterize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1).""" 
        eps = to_var(torch.randn(mu.size(0), mu.size(1), mu.size(2)))
        z = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std  
        return z                                                                                                                                                           
 
    def forward(self, word_attention_vectors, state_sent):
        z, state_sent = self.sent_gru(word_attention_vectors, state_sent)
        #print "Size of z", z.size()
        sent_mu, sent_log_var = torch.chunk(z, 2, dim=2)
        #print "Size of sentmu, sent_lv", sent_mu.size(), sent_log_var.size()
        output_sent = self.reparameterize(sent_mu, sent_log_var)        
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent,self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn.transpose(1,0))
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1,0))
        final_map = self.final_linear(sent_attn_vectors.squeeze(0))
        if self.return_softmax:
            return F.softmax(final_map), state_sent, sent_attn_norm, sent_mu, sent_log_var
        else:
            return final_map, state_sent, sent_attn_norm, sent_mu, sent_log_var

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            if self.bidirectional == True:
                return Variable(torch.zeros(2, self.batch_size, 2*self.sent_gru_hidden))
            else:
                return Variable(torch.zeros(1, self.batch_size, 2*self.sent_gru_hidden))        
        else:
            if self.bidirectional == True:
                return Variable(torch.zeros(2, batch_size, 2*self.sent_gru_hidden))
            else:
                return Variable(torch.zeros(1, batch_size, 2*self.sent_gru_hidden))        

# ## Functions to train the model

# In[7]:

class CustomHDAMVAE(nn.Module):
    def __init__(self, batch_size=64, num_tokens=100000, embed_size=300,
                 word_gru_hidden=100, bidirectional= True,
                 sent_gru_hidden=100, n_classes=3, return_softmax=False):
        super(CustomHDAMVAE, self).__init__()
        self.batch_size=batch_size
        self.num_tokens=num_tokens
        self.embed_size=embed_size
        self.word_gru_hidden=word_gru_hidden
        self.bidirectional=bidirectional
        self.sent_gru_hidden=sent_gru_hidden
        self.n_classes=n_classes
        self.return_softmax=return_softmax
        
        self.word_attn_model = AttentionWordRNN(batch_size=self.batch_size, num_tokens=self.num_tokens, 
                                              embed_size=self.embed_size, 
                                              word_gru_hidden=self.word_gru_hidden, bidirectional= self.bidirectional)
        
        self.sent_attn_model = AttentionSentRNNVAE(batch_size=self.batch_size, sent_gru_hidden=self.sent_gru_hidden, 
                                          word_gru_hidden=self.word_gru_hidden, 
                                        n_classes=self.n_classes, bidirectional= self.bidirectional, return_softmax=False) # We will return softmax if needed
        
    def forward(self, mini_batch):
        max_sents, batch_size, max_tokens = mini_batch.size()
        state_word = self.word_attn_model.init_hidden(batch_size).cuda()
        state_sent = self.sent_attn_model.init_hidden(batch_size).cuda()
        s = None
        for i in xrange(max_sents):
            _s, state_word, _ = self.word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)            
        y_pred, state_sent, _ , sent_mu, sent_log_var= self.sent_attn_model(s, state_sent)
        if self.return_softmax:
            return F.log_softmax(y_pred), sent_mu, sent_log_var
        else:
            return y_pred, sent_mu, sent_log_var


class CustomHDAM(nn.Module):
    def __init__(self, batch_size=64, num_tokens=100000, embed_size=300,
                 word_gru_hidden=100, bidirectional= True,
                 sent_gru_hidden=100, n_classes=3, return_softmax=False):
        super(CustomHDAM, self).__init__()
        self.batch_size=batch_size
        self.num_tokens=num_tokens
        self.embed_size=embed_size
        self.word_gru_hidden=word_gru_hidden
        self.bidirectional=bidirectional
        self.sent_gru_hidden=sent_gru_hidden
        self.n_classes=n_classes
        self.return_softmax=return_softmax
        
        self.word_attn_model = AttentionWordRNN(batch_size=self.batch_size, num_tokens=self.num_tokens, 
                                              embed_size=self.embed_size, 
                                              word_gru_hidden=self.word_gru_hidden, bidirectional= self.bidirectional)
        
        self.sent_attn_model = AttentionSentRNN(batch_size=self.batch_size, sent_gru_hidden=self.sent_gru_hidden, 
                                          word_gru_hidden=self.word_gru_hidden, 
                                        n_classes=self.n_classes, bidirectional= self.bidirectional, return_softmax=False) # We will return softmax if needed
        
    def forward(self, mini_batch):
        max_sents, batch_size, max_tokens = mini_batch.size()
        state_word = self.word_attn_model.init_hidden(batch_size).cuda()
        state_sent = self.sent_attn_model.init_hidden(batch_size).cuda()
        s = None
        for i in xrange(max_sents):
            _s, state_word, _ = self.word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)            
        y_pred, state_sent, _ = self.sent_attn_model(s, state_sent)
        if self.return_softmax:
            return F.log_softmax(y_pred)
        else:
            return y_pred


def get_predictions(val_tokens, hadm_model):
    return hadm_model(val_tokens)

def pad_batch(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.mean([len(x) for x in mini_batch]))
    max_token_len = int(np.mean([len(val) for sublist in mini_batch for val in sublist]))
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
    for i in xrange(main_matrix.shape[0]):
        for j in xrange(main_matrix.shape[1]):
            for k in xrange(main_matrix.shape[2]):
                try:
                    main_matrix[i,j,k] = mini_batch[i][j][k]
                except IndexError:
                    pass
    return Variable(torch.from_numpy(main_matrix).transpose(0,1))


def test_accuracy_mini_batch(tokens, labels, hadm_model):
    from sklearn.metrics import f1_score
    y_pred = get_predictions(tokens, hadm_model)
    _, y_pred = torch.max(y_pred, 1)
    correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
    labels = np.ndarray.flatten(labels.data.cpu().numpy())
    return f1_score(labels, correct, average='weighted')



def test_accuracy_full_batch(tokens, labels, mini_batch_size, hadm_model):
    from sklearn.metrics import f1_score
    p = []
    l = []
    g = gen_minibatch(tokens, labels, mini_batch_size)
    for token, label in g:
        y_pred = get_predictions(token.cuda(), hadm_model)
        _, y_pred = torch.max(y_pred, 1)
        p.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
        l.append(np.ndarray.flatten(label.data.cpu().numpy()))
    p = [item for sublist in p for item in sublist]
    l = [item for sublist in l for item in sublist]
    p = np.array(p)
    l = np.array(l)
    return f1_score(l, p, average='weighted')


def test_data(mini_batch, targets, hadm_model): 
    y_pred=hadm_model(mini_batch)
    loss = criterion(y_pred.cuda(), targets)     
    return loss.data[0]



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def gen_minibatch(tokens, labels, mini_batch_size, shuffle= True):
    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
        token = pad_batch(token)
        yield token.cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda()    


def check_val_loss(val_tokens, val_labels, mini_batch_size, hadm_model):
    val_loss = []
    for token, label in iterate_minibatches(val_tokens, val_labels, mini_batch_size, shuffle= True):
        val_loss.append(test_data(pad_batch(token).cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda(), 
                                  hadm_model))
    return np.mean(val_loss)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train_early_stopping(mini_batch_size, X_train, y_train, X_test, y_test, hadm_model, 
                         optimizer, loss_criterion, num_epoch, 
                         print_val_loss_every = 1000, print_loss_every = 50):
    start = time.time()
    loss_full = []
    loss_epoch = []
    accuracy_epoch = []
    loss_smooth = []
    accuracy_full = []
    epoch_counter = 0
    g = gen_minibatch(X_train, y_train, mini_batch_size)
    for i in xrange(1, num_epoch + 1):
        try:
            optimizer.zero_grad()
            tokens, labels = next(g)
            y_pred=hadm_model(tokens)
            loss = criterion(y_pred.cuda(), labels)
            loss.backward()
            optimizer.step()
            acc = test_accuracy_mini_batch(tokens, labels, hadm_model)
            accuracy_full.append(acc)
            accuracy_epoch.append(acc)
            loss_full.append(loss.data[0])
            loss_epoch.append(loss.data[0])
            # print loss every n passes
            if i % print_loss_every == 0:
                accuracy_epoch.append(acc)
                print 'Loss at %d minibatches, %d epoch,(%s) is %f' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch))
                print 'Accuracy at %d minibatches is %f' % (i, np.mean(accuracy_epoch))
             #check validation loss every n passes
            if i % print_val_loss_every == 0:
                val_loss = check_val_loss(X_test, y_test, mini_batch_size, hadm_model)
                print 'Average training loss at this epoch..minibatch..%d..is %f' % (i, np.mean(loss_epoch))
                print 'Validation loss after %d passes is %f' %(i, val_loss)
                if val_loss > np.mean(loss_full):
                    print 'Validation loss is higher than training loss at %d is %f , stopping training!' % (i, val_loss)
                    print 'Average training loss at %d is %f' % (i, np.mean(loss_full))
        except StopIteration:
            epoch_counter += 1
            print 'Reached %d epocs' % epoch_counter
            print 'i %d' % i
            g = gen_minibatch(X_train, y_train, mini_batch_size)
            loss_epoch = []
            accuracy_epoch = []
    return loss_full
