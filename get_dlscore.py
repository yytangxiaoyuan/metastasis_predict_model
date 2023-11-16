# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
import os
from os.path import join as pjoin  # pylint: disable=g-importing-member
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score,confusion_matrix
from Delong import delong_roc_variance
import scipy.stats as stats
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

import models as models
from Data.MyDataset1 import MyDataset

import bit_hyperrule

def mktrainval(args):
    """Returns train and validation datasets."""
    precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
    val_tx = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,)),
    ])

    train_set = MyDataset(pjoin(args.datadir, "train"), transforms=val_tx)###
    test_set = MyDataset(pjoin(args.datadir, "test"), transforms=val_tx)###

    print(f"Using a training set with {len(train_set)} images.")
    print(f"Using a test set with {len(test_set)} images.")

    micro_batch_size = args.batch

    # 设置pin_memory为真的时候，数据转到GPU显存的速度可以更快
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    if micro_batch_size <= len(train_set):
      train_loader = torch.utils.data.DataLoader(
          train_set, batch_size=micro_batch_size, shuffle=True,
          num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
      # In the few-shot cases, the total dataset size might be smaller than the batch-size.
      # In these cases, the default sampler doesn't repeat, so we need to make it do that
      # if we want to match the behaviour from the paper.
      train_loader = torch.utils.data.DataLoader(
          train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
          sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))
    
    return train_set, train_loader, test_set, test_loader

def Delong(outcome,pred):
    alpha = .95
    auc, auc_cov = delong_roc_variance(outcome,pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std); ci[ci > 1] = 1
    return ci

def run_eval(model, data_loader, device):
  # switch to evaluate mode
  model.eval()
  print("Running validation...")
  
  temp = nn.Softmax(dim = 1)
  result,initial = [],[]
  acc = 0
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # compute output, measure accuracy and record loss.
        logits = model(x)
        accuracy = (torch.argmax(logits,1) == y).sum().float()
        predicted = temp(logits)
        for i in range(len(y)):
            result.append(predicted[i,1].detach().cpu().numpy()) 
        initial.append(y.detach().cpu().numpy())
        acc += accuracy.item()
        # print(len(initial))
  result = [i for i in result]
  initial = [i for j in initial for i in j]
  auc = roc_auc_score(np.array(initial), np.array(result))
  predicted_label = result.copy(); 
  for i in range(len(predicted_label)):
      if predicted_label[i] < 0.5:
          predicted_label[i] = 0
      else:
          predicted_label[i] = 1
  tn, fp, fn, tp = confusion_matrix(np.array(initial), np.array(predicted_label)).ravel()
  sens = tp/(tp+fn)
  spec = tn/(fp+tn)
  ppv = tp/(tp+fp)
  npv = tn/(tn+fn)
  mcc = (tp*tn-tp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
  f1score = 2*(ppv*npv)/(ppv+npv)
  ci = Delong(np.array(initial), np.array(result))
  acc /= len(data_loader.dataset)
  print(f'acc: {acc:.4f},\t auc: {auc:.4f}, 95%CI: {ci}')
  print(f'sens: {sens:.4f},\t spec: {spec:.4f},\t ppv: {ppv:.4f},\t npv: {npv:.4f}')
  print(f'mcc: {mcc:.4f},\t f1score: {f1score:.4f}')
  
  temp = 1-np.array(result)
  predict = np.transpose(np.vstack((temp,np.array(result))))
  return np.array(initial), predict

def main(args):
    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True # 在网络结构固定的情况下，可以根据每个卷积层搜索最适合的卷积实现算法，从而加快运行速度

    # set the running device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available() and args.gpu != '-1' and args.gpu != 'cpu':
        device = torch.device('cuda')
    if args.gpu == '-1' or args.gpu == 'cpu':
        device = torch.device('cpu')
    
    # perpare the training data
    train_set, train_loader, test_set, test_loader = mktrainval(args)

    print(f"Loading model from {args.model}.npz")
    model = models.KNOWN_MODELS[args.model](head_size=2, zero_head=True)
    model.load_from(np.load(f"/home/NeverDie/code/bit_pytorch/{args.model}.npz"))
    
    if args.multiGPU == True:
        model = nn.DataParallel(model)

    # Resume fine-tuning if we find a saved model.
    checkpoint = torch.load(args.modeldir, map_location="cpu")
    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])  # module.
    print(f"Resumed at step {step}")
    model = model.to(device)
    
    OutcomeTrain, ResultTrain = run_eval(model, train_loader, device)
    OutcomeTest, ResultTest = run_eval(model, test_loader, device)
    # np.save('/home/NeverDie/code/bit_pytorch/AllPredictResults/DlResult_train.npy',ResultTrain)
    # np.save('/home/NeverDie/code/bit_pytorch/AllPredictResults/DlResult_test.npy',ResultTest)
    # np.save('/home/NeverDie/code/bit_pytorch/AllPredictResults/DlOutcome_train.npy',OutcomeTrain)
    # np.save('/home/NeverDie/code/bit_pytorch/AllPredictResults/DlOutcome_test.npy',OutcomeTest)
    print('Finish evaluating.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu",type=str,default='3',help="gpu id")
    parser.add_argument("--multiGPU",type=bool,dest="multiGPU", 
                        default=False,help="whether use multi GPU training")
    parser.add_argument("--datadir", default='/home/NeverDie/GaoData/',
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--model", choices=list(models.KNOWN_MODELS.keys()),
                        default='BiT-M-R50x1',
                        help="Which variant to use; BiT-M gives best results.")
    parser.add_argument("--modeldir", type=str, dest="modeldir", 
                        default="/home/NeverDie/code/bit_pytorch/33.pth.tar", help="models folder")###
    parser.add_argument("--dataset", choices=list(bit_hyperrule.known_dataset_sizes.keys()),
                        default='Gao3')  
    main(parser.parse_args())
