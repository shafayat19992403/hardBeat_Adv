import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn.functional as F

from torchvision.utils import save_image

from hardbeat import hardbeat
from util import get_loader, get_data, get_model
from util import get_classes, get_norm, get_size


def eval_acc(model, loader, device):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


def test(args):
    print('Loading model...')
    

    model = get_model(args.dataset, args.network).to(device)

    #model = get_model(args.dataset, args.network).to(args.device)
    if args.dataset in ['stl10', 'svhn', 'gtsrb']:
        path = f'ckpt/{args.dataset}_{args.network}.pt'
        model = torch.load(path, map_location=args.device)
    model = torch.nn.DataParallel(model)
    model.eval()

    print('Testing...')
    test_loader = get_loader(args.dataset, False, args.batch_size)
    acc = eval_acc(model, test_loader, args.device)
    print(f'ACC: {acc:.4f}')


def train(args):
    model = get_model(args.dataset, args.network).to(args.device)
    model = torch.nn.DataParallel(model)

    train_loader = get_loader(args.dataset, True,  args.batch_size)
    test_loader  = get_loader(args.dataset, False, args.batch_size)

    criterion = torch.nn.CrossEntropyLoss()

    if 'vgg' in args.network:
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                    gamma=0.5)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9,
                                    weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                    gamma=0.1)

    save_path = f'ckpt/{args.dataset}_{args.network}.pt'

    best_acc = 0
    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        scheduler.step()
        time_end = time.time()
        acc = eval_acc(model, test_loader, args.device)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, '
                         .format(epoch, step, time_end-time_start) +\
                         'loss: {:.4f}, acc: {:.4f}\n'.format(loss, acc))
        sys.stdout.flush()
        time_start = time.time()

        if acc > best_acc:
            best_acc = acc
            print(f'--- BEST ACC: {best_acc:.4f} ---')
            torch.save(model.module, save_path)



def attack(args):
    source, target = list(map(int, args.pair.split('-')))

    input_size = get_size(args.dataset)
    num_classes = get_classes(args.dataset)
    normalize, unnormalize = get_norm(args.dataset)

    print('Loading model...')
    model = get_model(args.dataset, args.network).to(args.device)
    if args.dataset in ['stl10', 'svhn', 'gtsrb']:
        path = f'ckpt/{args.dataset}_{args.network}.pt'
        model = torch.load(path, map_location=args.device)
    model.eval()

    print('Loading data...')
    gen_loader  = get_loader(args.dataset, False, args.batch_size)
    test_loader = get_loader(args.dataset, False, args.batch_size)

    x_gen, y_gen = get_data(gen_loader, source, args.attack_size)
    x_target, y_target = get_data(gen_loader, target, 1)

    x_gen = unnormalize(x_gen)[:args.attack_size]
    y_gen = y_gen[:args.attack_size]

    print('Attacking...')
    trigger_path = f'data/trigger/{args.dataset}'
    if not os.path.exists(trigger_path):
        os.makedirs(trigger_path)

    prefix = f'{args.dataset}_{args.network}'\
                + f'_{args.patch_size}x{args.patch_size}_{args.pair}'
    trigger_path = os.path.join(trigger_path, prefix)

    if args.log:
        if not os.path.exists('data/logs'):
            os.makedirs('data/logs')

        log_path = f'data/logs/{args.dataset}_{args.network}'\
                    + f'_{args.patch_size}x{args.patch_size}.txt'
        log = open(log_path, 'a')

    if args.load:
        mask    = torch.load(f'{trigger_path}_mask.pt')
        pattern = torch.load(f'{trigger_path}_pattern.pt')
        print('pixel num:\t', torch.count_nonzero(mask).numpy() / 3)
    else:
        perturbed, mask, pattern\
                = hardbeat(model, x_gen, normalize, target, pos=None,
                           size=args.patch_size, query_limit=args.query_limit)
        torch.save(mask,    f'{trigger_path}_mask.pt')
        torch.save(pattern, f'{trigger_path}_pattern.pt')

        if args.save:
            if not os.path.exists('data/images'):
                os.makedirs('data/images')

            for i in range(min(len(perturbed), 10)):
                image = torch.cat((x_gen[i], torch.zeros((3, input_size[0], 8)),
                                   perturbed[i]), dim=2)
                save_image(image, f'data/images/{prefix}_{i}.png')

    print('Evaluating...')
    total = 0
    correct = 0
    for i, (x_batch, y_batch) in enumerate(test_loader):
        indices = np.where(y_batch == source)[0]
        if len(indices) <= 0:
            continue

        x_batch = unnormalize(x_batch[indices])
        x_batch = (1 - mask) * x_batch + mask * pattern
        x_batch = torch.clamp(x_batch, 0, 1)

        pred = model(normalize(x_batch).to(args.device)).max(dim=1)[1]
        compare = torch.eq(pred, target).type(torch.FloatTensor).sum()
        total += len(pred)
        correct += compare.numpy()
    asr = correct / total
    print('Test ASR:', asr)

    if args.log:
        log.write(f'{asr},')
        if f'-{num_classes-1}' in args.pair:
            log.write('\n')
        log.close()

def adversarial_train(args):
    print('is inside')
    source, target = list(map(int, args.pair.split('-')))
    model = get_model(args.dataset, args.network).to(args.device)
    model = torch.nn.DataParallel(model)

    train_loader = get_loader(args.dataset, True,  args.batch_size)
    test_loader  = get_loader(args.dataset, False, args.batch_size)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    save_path = f'ckpt/{args.dataset}_{args.network}_adv_trained.pt'

    best_acc = 0
    for epoch in range(args.epochs_adv):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)

            # Generate adversarial examples (for a subset or all)
            #x_batch_adv, _, _ = hardbeat(model.module, x_batch, get_norm(args.dataset)[0], target=y_batch, size=args.patch_size, query_limit=args.query_limit)
            x_batch_adv, _, _ = hardbeat(model, x_batch, get_norm(args.dataset)[0], target, size=args.patch_size, query_limit=args.query_limit)
            optimizer.zero_grad()
            output_clean = model(x_batch)
            output_adv = model(x_batch_adv)

            loss_clean = criterion(output_clean, y_batch)
            loss_adv = criterion(output_adv, y_batch)
            loss = loss_clean + loss_adv  # Combine losses

            loss.backward()
            optimizer.step()

            #pred = output_clean.max(dim=1)[1]

            # if step % 10 == 0:
            print(f'Epoch {epoch}, Step {step}, Loss Clean: {loss_clean.item()}, Loss Adv: {loss_adv.item()}')

        scheduler.step()
        acc = eval_acc(model, test_loader, args.device)
        print(f'Epoch {epoch}, ACC: {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            print(f'--- BEST ACC: {best_acc:.4f} ---')
            torch.save(model.module.state_dict(), save_path)

def evaluate_on_adversarial_test_set(model, test_loader, args):
    print('Evaluating on adversarial examples...')
    adv_acc = eval_acc(model, test_loader, args.device)  # Use an appropriate loader that provides adversarial examples
    print(f'Adversarial ACC: {adv_acc:.4f}')

###############################################################################
############                          main                         ############
###############################################################################
# def main(args):
#     if args.phase == 'test':
#         test(args)
#     elif args.phase == 'train':
#         train(args)
#     elif args.phase == 'attack':
#         attack(args)
#     elif args.phase == 'adv_train':
#         adversarial_train(args)
#     else:
#         print('Option [{}] is not supported!'.format(args.phase))

def main(args):
    print('inside main')
    if args.phase == 'test':
        print('inside test')
        test(args)
    elif args.phase == 'train':
        print('inside train')
        train(args)
        # Evaluate on adversarial examples after regular training
        print("\nEvaluating on adversarial test set after regular training...")
        #evaluate_on_adversarial_test_set(model, test_loader, args)
    elif args.phase == 'attack':
        print('inside attack')
        attack(args)
    elif args.phase == 'adv_train':
        print('inside adv_train')
        adversarial_train(args)
        # Reload the best model saved by adversarial_train
        #model.load_state_dict(torch.load(f'ckpt/{args.dataset}_{args.network}_adv_trained.pt', map_location=args.device))
        # Evaluate on adversarial examples after adversarial training
        #print("\nEvaluating on adversarial test set after adversarial training...")
        #evaluate_on_adversarial_test_set(model, test_loader, args)
    else:
        print('Option [{}] is not supported!'.format(args.phase))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--gpu',     default='7',        help='gpu id')
    parser.add_argument('--phase',   default='attack',   help='phase of framework')
    parser.add_argument('--dataset', default='cifar10',  help='dataset')
    parser.add_argument('--network', default='resnet18', help='model structure')
    parser.add_argument('--pair',    default='0-1',      help='label pair')

    parser.add_argument('--seed',        default=714725708, type=int, help='random seed')
    parser.add_argument('--epochs',      default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size',  default=128, type=int, help='batch size')
    parser.add_argument('--attack_size', default=100, type=int, help='attack size')
    parser.add_argument('--patch_size',  default=7,   type=int, help='patch size')
    parser.add_argument('--query_limit', default=50000, type=int, help='number of queries')

    parser.add_argument('--load', action='store_true', help='load generated trigger')
    parser.add_argument('--save', action='store_true', help='save images')
    parser.add_argument('--log',  action='store_true', help='log results')


    parser.add_argument('--adv_train', action='store_true', help='Enable adversarial training')
    parser.add_argument('--epochs_adv', default=5, type=int, help='number of epochs for adversarial training')

    args = parser.parse_args()
    args.device = torch.device('cuda')
    # Dynamically setting device based on CUDA availability
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = device

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_start = time.time()
    main(args)
    time_end = time.time()
    print('='*50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('='*50)



