import os
import sys
import json
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from data.Indoor3DSemSegLoader import Indoor3DSemSeg


sys.path.append(".")
from lib.solver import Solver
from lib.dataset import ScannetDataset, ScannetDatasetWholeScene, collate_random, collate_wholescene
from lib.loss import WeightedCrossEntropyLoss
from lib.config import CONF


input_folder = "/home/lorenzlamm/Dokumente/DavesPointnet/Pointnet2.ScanNet/data"

def get_dataloader(args, scene_list, is_train=True, is_wholescene=False):
    if is_wholescene:
        dataset = ScannetDatasetWholeScene(scene_list, is_train=is_train)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene)
    else:
        dataset = ScannetDataset(scene_list, is_train=is_train)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_random)

    return dataset, dataloader

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader, stamp, weight, is_wholescene):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pointnet2/'))
    Pointnet = importlib.import_module("pointnet2_msg_semseg")

    model = Pointnet.get_model(num_classes=21).cuda()
    num_params = get_num_params(model)
    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    solver = Solver(model, dataloader, criterion, optimizer, args.batch_size, stamp, is_wholescene)

    return solver, num_params

def get_scene_list(path):
    scene_list = []
    with open(path) as f:
        for scene_id in f.readlines():
            scene_list.append(scene_id.strip())

    return scene_list

def save_info(args, root, train_examples, val_examples, num_params):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = train_examples
    info["num_val"] = val_examples
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def train(args):
    # init training dataset
    print("preparing data...")
    if args.debug:
        train_scene_list = ["scene0000_00"]
        val_scene_list = ["scene0000_00"]
    else:
        print("HI")
        #train_scene_list = get_scene_list(CONF.SCANNETV2_TRAIN)
        #val_scene_list = get_scene_list(CONF.SCANNETV2_VAL)

    # dataloader
    if args.wholescene:
        is_wholescene = True
    else:
        is_wholescene = False
    train_dataset = Indoor3DSemSeg(4096, root=input_folder, train=True)
    val_dataset = Indoor3DSemSeg(4096, root=input_folder, train=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=2,
        shuffle=True,
    )
#    train_dataset, train_dataloader = get_dataloader(args, train_scene_list, True, is_wholescene)
#    val_dataset, val_dataloader = get_dataloader(args, val_scene_list, True, is_wholescene)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }
    weight = train_dataset.labelweights
    train_examples = len(train_dataset)
    val_examples = len(val_dataset)

    print("initializing...")
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = os.path.join(CONF.OUTPUT_ROOT, stamp)
    os.makedirs(root, exist_ok=True)
    solver, num_params = get_solver(args, dataloader, stamp, weight, is_wholescene)
    
    print("\n[info]")
    print("Train examples: {}".format(train_examples))
    print("Evaluation examples: {}".format(val_examples))
    print("Start training...\n")
    save_info(args, root, train_examples, val_examples, num_params)
    solver(args.epoch, args.verbose)

def predForVisualization(args):
    test_dataset = Indoor3DSemSeg(16384*2, root="/home/lorenzlamm/Dokumente/DavesPointnetClone/Pointnet2.ScanNet/preprocessing/scannet_scenes", train=False)

    weight = test_dataset.labelweights
    train_examples = len(test_dataset)

    print("initializing...")
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = os.path.join(CONF.OUTPUT_ROOT, stamp)
    os.makedirs(root, exist_ok=True)

    model_path = os.path.join(CONF.OUTPUT_ROOT, args.folder, "model.pth")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pointnet2/'))
    Pointnet = importlib.import_module("pointnet2_msg_semseg")
    model = Pointnet.get_model(num_classes=21).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    sceneCount = 0
    for i in range(len(test_dataset)):
        coords, feats, semantic_segs, sample_weights, fetch_time = test_dataset[i]
        coords, feats, semantic_segs, sample_weights = torch.tensor(coords).cuda(), torch.tensor(
            feats).cuda(), torch.tensor(semantic_segs).cuda(), torch.tensor(sample_weights).cuda()
        output = model(coords.unsqueeze(0))
        preds = torch.argmax(output, dim=2)
        preds = torch.cuda.FloatTensor(preds.float())
        if (sceneCount == 0):
            print(coords.squeeze(0).shape)
            print(torch.cuda.FloatTensor(preds.squeeze().unsqueeze(1)).shape)
            all_points = torch.cat((coords.squeeze(0), torch.cuda.FloatTensor(preds.squeeze().unsqueeze(1))), dim=1)
        else:
            all_points = torch.cat((all_points, torch.cat((coords.squeeze(0), torch.cuda.FloatTensor(preds.squeeze().unsqueeze(1))), dim=1)), dim=0)
        print(all_points.shape)
        sceneCount += 1

    unique_points = torch.unique(all_points, dim=-2)
    print(unique_points.shape)
    final_points = unique_points.clone()
    for i in range(unique_points.shape[0]):
        point = unique_points[i, :3]
        mask = torch.sum(all_points[:, :3] == point, dim=1)
        pointMask = mask == 3
        points = all_points[pointMask]
        point_counts = torch.zeros(21)
        for point in points:
            for j in range(21):
                if (point[3] == j):
                    point_counts[j] += 1
        final_points[i, 3] = torch.argmax(point_counts)
        if (i % 300 == 0):
            print(i)
    torch.save(final_points, "/home/lorenzlamm/Dokumente/DavesPointnetClone/Pointnet2.ScanNet/preprocessing/scannet_scenes/final_scene.torch")
    write_torch_to_text("/home/lorenzlamm/Dokumente/DavesPointnetClone/Pointnet2.ScanNet/preprocessing/scannet_scenes/final_scene.torch", "/home/lorenzlamm/Dokumente/DavesPointnetClone/Pointnet2.ScanNet/preprocessing/scannet_scenes/final_scene.txt")
    write_np_to_text("/home/lorenzlamm/Dokumente/DavesPointnetClone/Pointnet2.ScanNet/preprocessing/scannet_scenes/whole_scene/scene0700_00.npy", "/home/lorenzlamm/Dokumente/DavesPointnetClone/Pointnet2.ScanNet/preprocessing/scannet_scenes/scene0700_00.txt")

def write_torch_to_text(filename, outname):
    file = torch.load(filename).cpu().numpy()
    with open(outname, "w+") as f:
        for i in range(file.shape[0]):
            f.write(str(file[i][0]) + " " + str(file[i][1]) + " " + str(file[i][2]) + " " + str(file[i][3]) + "\n")

def write_np_to_text(filename, outname, index=7):
    file = np.load(filename)
    with open(outname, "w+") as f:
        for i in range(file.shape[0]):
            f.write(str(file[i][0]) + " " + str(file[i][1]) + " " + str(file[i][2]) + " " + str(file[i][index]) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument('--batch_size', type=int, help='batch size', default=2)
    parser.add_argument('--epoch', type=int, help='number of epochs', default=10)
    parser.add_argument('--verbose', type=int, help='iterations of showing verbose', default=1)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-3)
    parser.add_argument('--wd', type=float, help='weight decay', default=0)
    parser.add_argument('--bn', type=bool, help='batch norm', default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wholescene", action="store_true")
    args = parser.parse_args()
    args.folder = "/home/lorenzlamm/Dokumente/DavesPointnetClone/Pointnet2.ScanNet"

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    #train(args)
    predForVisualization(args)