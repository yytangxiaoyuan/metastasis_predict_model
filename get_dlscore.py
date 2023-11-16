import os
from os.path import join as pjoin 
from argparse import ArgumentParser
import scipy.stats as stats
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

import models as models
from MyDataset1 import MyDataset

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

    train_set = MyDataset(pjoin(args.datadir, "train"), transforms=val_tx)
    test_set = MyDataset(pjoin(args.datadir, "valid"), transforms=val_tx)

    print(f"Using a training set with {len(train_set)} images.")
    print(f"Using a test set with {len(test_set)} images.")

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    return train_loader, test_loader


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

        logits = model(x)
        accuracy = (torch.argmax(logits,1) == y).sum().float()
        predicted = temp(logits)
        for i in range(len(y)):
            result.append(predicted[i,1].detach().cpu().numpy()) 
        initial.append(y.detach().cpu().numpy())
  result = [i for i in result]
  initial = [i for j in initial for i in j]
  temp = 1-np.array(result)
  predict = np.transpose(np.vstack((temp,np.array(result))))
  return np.array(initial), predict

def main(args):
    torch.backends.cudnn.benchmark = True

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
    model.load_from(np.load(f"your_path/{args.model}.npz"))
    model = nn.DataParallel(model)

    # Resume fine-tuning if we find a saved model.
    checkpoint = torch.load(args.modeldir, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    
    OutcomeTrain, ResultTrain = run_eval(model, train_loader, device)
    OutcomeTest, ResultTest = run_eval(model, test_loader, device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu",type=str,default='3',help="gpu id")
    parser.add_argument("--datadir", require=True,
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--batch", type=int, default=256, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of background threads used to load data.")
    parser.add_argument("--model", choices=list(models.KNOWN_MODELS.keys()),
                        default='BiT-M-R50x1', help="Which variant to use; BiT-M gives best results.")
    parser.add_argument("--modeldir", type=str, require=True, help="models folder")
    parser.add_argument("--dataset", choices=list(bit_hyperrule.known_dataset_sizes.keys()),
                        require=True)  
    main(parser.parse_args())
