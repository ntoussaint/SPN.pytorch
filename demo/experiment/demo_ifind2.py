import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt


import torch
import torch.nn as nn
import torchvision.transforms as torchtransforms

from copy import deepcopy
from experiment.spn_models import *

# tensorboardX
from tensorboardX import SummaryWriter

# warnings
import warnings
warnings.filterwarnings('ignore', '.*deprecated.*')

# transforms
import torchbiomedvision.transforms.itk_transforms as itktransforms
import torchbiomedvision.transforms.tensor_transforms as tensortransforms

# dataset
from torchbiomedvision.datasets.itk_metadata_classification import ITKMetaDataClassification

# utils
from torchbiomedvision.utils import pytorch_utils as torchutils
from torchbiomedvision.utils import python_utils as utils
from torchbiomedvision.utils import engine as torchengine


def define_transforms(hp):

    keys=('spacing', 'aspect_ratio', 'image_size','rescale_interval','flip')
    for k in keys:
        assert(k in hp.keys()), 'hyper parameters should contain the key \'{}\''.format(k)

    ## create transformation and data augmentation schemes

    # spacing is arbitrary
    resample = itktransforms.Resample(new_spacing=hp['spacing'])
    # transform an ITK image into a numpy array
    tonumpy = itktransforms.ToNumpy(outputtype='float')
    # transform a numpy array into a torch tensor
    totensor = torchtransforms.ToTensor()
    # crop to an aspect ratio
    crop = tensortransforms.CropToRatio(outputaspect=hp['aspect_ratio'])
    # padd to an aspect ratio
    padd = tensortransforms.PaddToRatio(outputaspect=hp['aspect_ratio'])
    # resize image to fixed size
    resize = tensortransforms.Resize(size=hp['image_size'], interp='bilinear')
    # rescale tensor to  interval
    rescale = tensortransforms.Rescale(interval=hp['rescale_interval'])
    # flip image in the y axis
    flip = tensortransforms.Flip(axis=2) if hp['flip'] else None

    # transforms to apply when learning
    train_transform = torchtransforms.Compose(
                        [resample,
                         tonumpy,
                         totensor,
                         crop,
                         resize,
                         rescale,
                         flip])

    # transforms to apply when validating
    val_transform = torchtransforms.Compose(
                            [resample,
                             tonumpy,
                             totensor,
                             crop,
                             resize,
                             rescale])

    hp['train_transform'] = train_transform
    hp['val_transform'] = val_transform

    return hp


def define_loaders(hp):

    keys=('datadir', 'train_transform','val_transform', 'batch_size', 'num_workers')
    for k in keys:
        assert(k in hp.keys()), 'hyper parameters should contain the key \'{}\''.format(k)

    datadir = hp['datadir']
    train_transform = hp['train_transform']
    val_transform = hp['val_transform']

    # load datasets
    train_dataset = ITKMetaDataClassification(root=datadir, mode='train',
                                              transform=train_transform)
    val_dataset   = ITKMetaDataClassification(root=datadir, mode='validate',
                                              transform=val_transform)

    # estimate the samples' weights
    train_cardinality = train_dataset.get_class_cardinality()
    val_cardinality = val_dataset.get_class_cardinality()
    train_sample_weights = torch.from_numpy(train_dataset.get_sample_weights())
    val_sample_weights = torch.from_numpy(val_dataset.get_sample_weights())

    print('')
    print('train-dataset: ')
    for idx, c in enumerate(train_dataset.get_classes()):
        print('{}: \t{}'.format(train_cardinality[idx], c))
    print('')
    print('validate-dataset: ')
    for idx, c in enumerate(val_dataset.get_classes()):
        print('{}: \t{}'.format(val_cardinality[idx], c))
    print('')

    # class labels
    classes_train = train_dataset.get_classes()
    classes_val = val_dataset.get_classes()

    assert(classes_train == classes_val), 'classes differ between train and validation sets'
    classes = classes_train
    del classes_train, classes_val

    # create samplers weighting samples according to the occurence of their respective class
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sample_weights,
                                                                   int(np.min(train_cardinality)),
                                                                   replacement=True)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=hp['batch_size'], shuffle=False,
                                               num_workers=hp['num_workers'], sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=hp['batch_size'], shuffle=False,
                                             num_workers=hp['num_workers'])

    hp['train_loader'] = train_loader
    hp['val_loader'] = val_loader

    hp['classes'] = classes

    return hp

def define_logging(hp):

    keys=('coordinate_system','arch', 'learning_rate', 'batch_size', 'aspect_ratio')
    for k in keys:
        assert(k in hp.keys()), 'hyper parameters should contain the key \'{}\''.format(k)

    # define output log directory
    p='cs={}-m={}-lr={}-bs={}-spn={}-aspect={}'.format(hp['coordinate_system'],
                                                       hp['arch'],
                                                       hp['learning_rate'],
                                                       hp['batch_size'],
                                                       1,
                                                       hp['aspect_ratio']
                                                      )
    p=os.path.join('logs', p)
    hp['save_model_path'] = p
    hp['Logger'] = SummaryWriter(log_dir=p)
    return hp

def define_model(hp):

    keys=('arch', 'classes', 'learning_rate', 'momentum', 'weight_decay')
    for k in keys:
        assert(k in hp.keys()), 'hyper parameters should contain the key \'{}\''.format(k)

    print('asking for model: {}'.format(hp['arch']))
    num_classes = len(hp['classes'])

    model = None

    if   hp['arch'] == 'resnet18':
        model = resnet18_sp(num_classes, num_maps=512, in_channels=1)
    elif hp['arch'] == 'resnet34':
        model = resnet34_sp(num_classes, num_maps=512, in_channels=1)
    elif hp['arch'] == 'vgg13':
        model = vgg13_sp(num_classes, batch_norm=False, num_maps=512, in_channels=1)
    elif hp['arch'] == 'vgg13_bn':
        model = vgg13_sp(num_classes, batch_norm=True, num_maps=512, in_channels=1)
    elif hp['arch'] == 'vgg16':
        model = vgg16_sp(num_classes, batch_norm=False, num_maps=512, in_channels=1)
    elif hp['arch'] == 'vgg16_bn':
        model = vgg16_sp(num_classes, batch_norm=True, num_maps=512, in_channels=1)
    elif hp['arch'] == 'alexnet':
        model = alexnet_sp(num_classes, num_maps=512, in_channels=1)

    print(model)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hp['learning_rate'],
                                momentum=hp['momentum'],
                                weight_decay=hp['weight_decay'])
    hp['model'] = model
    hp['criterion'] = criterion
    hp['optimizer'] = optimizer

    return hp

def explore_dataset(hp):

    keys=('train_loader', 'classes', 'batch_size')
    for k in keys:
        assert(k in hp.keys()), 'hyper parameters should contain the key \'{}\''.format(k)

    train_loader = hp['train_loader']

    # show an image
    def imshow(img):
        npimg = img.numpy()
        return plt.imshow(np.transpose(npimg, (1, 2, 0)))

    dataiter = iter(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images, labels = dataiter.next()
        break
    # show a minibatch
    plt.figure(0)
    imshow(torchvision.utils.make_grid([im for im in images]))
    plt.show()
    print('   '+' '.join('%5s' % hp['classes'][np.argmax(labels[j].numpy())] for j in range(hp['batch_size'])))


def define_dictionaries(variables):
    engine_state = {
        'use_gpu': torch.cuda.is_available(),
        'evaluate': False,
        'start_epoch': 0,
        'max_epochs': 30,
        'epoch_step':[10,20],
        'maximize': True,
        'resume': None,
        'use_pb': True,
    }

    p = {
        'image_size': [224, 224],
        'spacing': [.5, .5, 1.],
        'flip': True,
        'rescale_interval': [0,1],
        'batch_size': 6,
        'num_workers': 8,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'arch': 'resnet18',
        'coordinate_system': 'cart',
        'aspect_ratio': 1.,
    }

    print('Initializing state...')
    p.update(engine_state)

    print('Updating variables...')
    p.update(variables)

    print('Defining transforms...')
    p = define_transforms(p)

    print('Defining dataset...')
    p = define_loaders(p)

    print('Defining model...')
    p = define_model(p)

    print('Defining logging...')
    p = define_logging(p)

    return p

def learning(hp):

    keys=('model', 'train_loader', 'val_loader', 'criterion', 'optimizer', 'save_model_path')
    for k in keys:
        assert(k in hp.keys()), 'hyper parameters should contain the key \'{}\''.format(k)

    model = hp['model']
    train_loader = hp['train_loader']
    val_loader = hp['val_loader']
    criterion = hp['criterion']
    optimizer = hp['optimizer']

    # instantiate a MultiLabelMAP engine
    engine = torchengine.MultiLabelMAPEngine(hp)
    print(hp['save_model_path'])
    # learn
    engine.learning(model, criterion, train_loader, val_loader, optimizer)


parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-s', '--epoch-step', type=int, default=10,
                    help='step between each learning rate decrease')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='network architecture (resnet18, vgg13_bn, vgg16_bn,)')

def main_ifind2():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    state = {'batch_size': args.batch_size,
             'max_epochs': args.epochs,
             'image_size': [args.image_size] * 2,
             'evaluate': args.evaluate,
             'resume': args.resume,
             'learning_rate': args.lr,
             'epoch_step': range(0, args.epochs, args.epoch_step),
             'momentum': args.momentum,
             'weight_decay':args.weight_decay,
             'datadir': args.data,
             'arch': args.arch,
             'save_model_path': 'logs/ifind2/'
             }

    state = define_dictionaries(state)
    learning(state)

if __name__ == '__main__':
    main_ifind2()
