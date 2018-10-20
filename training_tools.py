import torch
import torchtext
import revtok
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd
import visdom
import pandas as pd
import pandas as pd
import pylab as pl
import scikitplot
from scipy.stats import entropy
from sklearn.metrics import f1_score
use_cuda = torch.cuda.is_available()
vis=visdom.Visdom()
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, return_softmax=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.return_softmax = return_softmax
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        if self.return_softmax:
            return F.log_softmax(out)
        else:
            return out

class CustomNet(nn.Module):
    def __init__(self, input_size, hidden_size, return_softmax=False):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.return_softmax = return_softmax
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if self.return_softmax:
            return F.softmax(out)
        else:
            return out

def train(model, optimizer, train_dataset, model_dir, model_prefix, num_epochs, get_examples, get_targets, lr=0.01, max_norm=None, compute_metric=None, eval_data=None, plot_every=50, strict_batch=False):
    # Always feed examples batch first
    epoch_losses=[]
    metrics=[]
    training_metrics=[]
    mean_losses=[]
    train_dataset.iterations=0
    save_every=1
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    loss_function=nn.NLLLoss()
    for epoch in np.arange(0,num_epochs):
        batch_losses=[]
        mean_losses=[]
        for i, b in enumerate(train_dataset):
            if strict_batch and (b.batch_size != train_dataset.batch_size):
                continue
            model.train()
            model.zero_grad()
            output=model(get_examples(b))
            targets=get_targets(b)
            loss = loss_function(output, targets)
            batch_losses.append(loss.data[0])
            if (i%plot_every==0):
                mean_losses.append(np.mean(batch_losses))
                yvals=np.array(mean_losses)
                xvals=np.arange(0, len(yvals))
                vis.line(Y=yvals, X=xvals, win='batch_loss', opts={'title':'batch_loss'})
            loss.backward()
            if max_norm is not None:
                nn.utils.clip_grad_norm(parameters, max_norm=max_norm)   
            optimizer.step()
            
        epoch_losses.append(np.mean(batch_losses))
        if compute_metric is not None:
            tmetric=compute_metric(train_dataset, model, get_examples, get_targets)
            training_metrics.append(tmetric)
            vis.line(np.array(training_metrics), win='training_metric', opts={'title':'training_metric'})
            metric=compute_metric(eval_data, model, get_examples, get_targets)
            metrics.append(metric)
            vis.line(Y=np.array(metrics), X=np.arange(0, len(np.array(metrics))), win='metric', opts={'title':'metric'})
            vis.line(Y=np.array(epoch_losses), X=np.arange(0, len(np.array(metrics))), win='loss', opts={'title':'loss'})
        torch.save(model.state_dict(), "{}/{}_{}.dict".format(model_dir, model_prefix, int(epoch)))
    return epoch_losses[-1]


def load_dataset(train_file_name, val_file_name, test_file_name, INDEX, TEXT, TARGET, build_vocab=True, min_freq=5, use_pretrained=False, pretrained_vecs=None, batch_size=2):
    train_dataset=torchtext.data.TabularDataset(train_file_name, format='tsv', fields=[('index',INDEX), ('example', TEXT), ('target', TARGET)], skip_header=True)
    if build_vocab:
        if use_pretrained:
            TEXT.build_vocab(train_dataset, vectors=pretrained_vecs, min_freq=min_freq)
        else:
            TEXT.build_vocab(train_dataset, min_freq=min_freq)
    train_iterator=torchtext.data.BucketIterator(train_dataset, train=True, batch_size=batch_size, repeat=False, shuffle=True)
    val_dataset=torchtext.data.TabularDataset(val_file_name, format='tsv', fields=[('index',INDEX), ('example', TEXT), ('target', TARGET)], skip_header=True)
    val_iterator=torchtext.data.BucketIterator(val_dataset, batch_size=batch_size, train=False, repeat=False, shuffle=False)
    test_dataset=torchtext.data.TabularDataset(test_file_name, format='tsv', fields=[('index',INDEX), ('example', TEXT), ('target', TARGET)], skip_header=True)
    test_iterator=torchtext.data.BucketIterator(test_dataset,train=False, repeat=False, batch_size=batch_size, shuffle=False)
    return train_iterator, val_iterator, test_iterator 

def load_dataset_fake(train_file_name, val_file_name, test_file_name, INDEX, TEXT, TARGET, SOURCE, DOCID, build_vocab=True, min_freq=5, use_pretrained=False, pretrained_vecs=None, batch_size=2):
    train_dataset=torchtext.data.TabularDataset(train_file_name, format='tsv', fields=[('index',INDEX), ('example', TEXT), ('target', TARGET), ('source', SOURCE), ('docid',DOCID)], skip_header=True)
    if build_vocab:
        if use_pretrained:
            TEXT.build_vocab(train_dataset, vectors=pretrained_vecs, min_freq=min_freq)
        else:
            TEXT.build_vocab(train_dataset, min_freq=min_freq)
    train_iterator=torchtext.data.BucketIterator(train_dataset, train=True, batch_size=batch_size, repeat=False, shuffle=True)
    val_dataset=torchtext.data.TabularDataset(val_file_name, format='tsv', fields=[('index',INDEX), ('example', TEXT), ('target', TARGET), ('source', SOURCE), ('docid', DOCID)], skip_header=True)
    val_iterator=torchtext.data.BucketIterator(val_dataset, batch_size=batch_size, train=False, repeat=False, shuffle=False)
    test_dataset=torchtext.data.TabularDataset(test_file_name, format='tsv', fields=[('index',INDEX), ('example', TEXT), ('target', TARGET), ('source', SOURCE), ('docid', DOCID)], skip_header=True)
    test_iterator=torchtext.data.BucketIterator(test_dataset,train=False, repeat=False, batch_size=batch_size, shuffle=False)
    return train_iterator, val_iterator, test_iterator 

def get_preds_on(dataset_iterator, model, get_examples):
    results=[]
    model.training=False
    for e in dataset_iterator:
        model.zero_grad()
        output=model(get_examples(e))
        for pred, ix in zip(output, e.index):
            results.append((ix.cpu().data.numpy()[0], pred.cpu().data.numpy()))

    model.training=True
    return results

def get_f1_on(dataset_iterator, model, get_examples, get_targets):
    all_preds=[]
    all_targets=[]
    model.training=False
    for e in dataset_iterator:
        model.zero_grad()
        output=model(get_examples(e))
        classix=list(np.argmax(output.cpu().data.numpy(), axis=1))
        targets=get_targets(e).cpu().data.numpy()
        all_preds.extend(classix)
        all_targets.extend(targets)
    model.training=True
    return f1_score(all_targets, all_preds, average='weighted')
