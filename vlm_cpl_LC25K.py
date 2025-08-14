import argparse
import copy
from matplotlib import pyplot as plt
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import time
from sklearn import metrics
import numpy as np
from dataset.alb_dataset import Tumor_dataset, Tumor_dataset_val, Tumor_dataset_val_cls, get_loader
import pandas as pd
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
from evaluate_util import hungarian_evaluate
from sklearn.cluster import KMeans


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def Prompt_feature_consensus(args, feature_all, y_preds):
    cluster_learner = KMeans(n_clusters=args.num_class, init='k-means++', n_init='auto')
    cluster_learner.fit(feature_all)
    cluster_idxs = cluster_learner.predict(feature_all)
    cluster_pred = np.array(cluster_idxs, dtype=np.uint8)
    hungarian_results = hungarian_evaluate(torch.tensor(y_preds).cpu(), torch.tensor(cluster_pred).cpu())
    reordered_preds = hungarian_results['reordered_preds']
    return reordered_preds.numpy()==y_preds, reordered_preds

def cls_recall(args, pred_array, target):
    pred, gt = np.zeros((args.num_class,)), np.zeros((args.num_class,))
    for i in range(len(target)):
        gt[target[i]] += 1
        if target[i]==pred_array[i]:
            pred[target[i]] += 1
    return pred/gt

def get_files(data_csv):
    data = pd.read_csv(data_csv)
    data_name = data.iloc[:, 0]
    data_label = data.iloc[:, 1]
    data_label = np.array(data_label).astype(np.uint8)
    data_name = data_name.to_list()
    new_file = [{"img": img, "label": label} for img, label in zip(data_name, data_label)]
    return new_file

def get_arguments():
    parser = argparse.ArgumentParser(
        description="xxxx Pytorch implementation")
    parser.add_argument("--num_class", type=int, default=8, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=512, help="Train batch size")
    parser.add_argument("--num_workers", default=12)
    return parser.parse_args()

def cal_acc(y_pred, y_true):
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    print(f"Test Accuracy: {test_accuracy}, f1:{f1}, precision:{p}, recall:{r}")

def zero_shot_inference(args, train_eval_loader, model_biomedclip, model_plip, plip_processor):
    with torch.no_grad():
        pred_all, gt_all, prob_all = torch.zeros((1, )), torch.zeros((1, )), torch.zeros((1, args.num_class))
        # embeddings = torch.zeros((1, 768))
        embeddings = torch.zeros((1, 512))
        # embeddings = torch.zeros((1, 1024))
        names = []
        for counter, sample in enumerate(train_eval_loader):
            x_batch = sample['img'].cuda()
            y_batch = sample['cls_label'].cuda()
            batch_names = sample['img_name']

            if counter == 0:
                print(batch_names[0])

            # biomedclip
            # image_features, text_features, logit_scale = model_biomedclip(x_batch, texts)
            # probs = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            # plip
            inputs = plip_processor(text=text_prompt, return_tensors="pt", padding=True)
            inputs['pixel_values'] = x_batch

            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            outputs = model_plip.forward(**inputs)
            # this is the image-text similarity score
            logits_per_image = outputs.logits_per_image
            probs_plip = logits_per_image.softmax(dim=1)

            # average
            # probs = (probs+probs_plip)/2
            probs = probs_plip

            logits_hard = torch.argmax(probs, dim=1)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
            gt_all = torch.cat((gt_all, y_batch.cpu()), dim=0)
            prob_all = torch.cat((prob_all, probs.cpu()), dim=0)
            names += batch_names
            # embeddings = torch.cat((embeddings, torch.cat((outputs.image_embeds.cpu(), image_features.cpu()), dim=1)), dim=0)
            embeddings = torch.cat((embeddings, outputs.image_embeds.cpu()), dim=0)

    pred_all, gt_all, embeddings, prob_all = pred_all[1:], gt_all[1:], embeddings[1:], prob_all[1:]
    y_true, y_pred = gt_all.numpy().astype(np.uint8), pred_all.numpy().astype(np.uint8)

    return y_pred, y_true, np.array(names), embeddings.clone().detach().cpu(), prob_all

def MVC(args, train_loader, model_biomedclip, model_plip, plip_processor, n_times):
    dropout_n = n_times # 5 is ok
    names = []
    with torch.no_grad():
        for j in range(dropout_n):
            pred, gt = np.zeros((args.num_class,)), np.zeros((args.num_class,))
            pred_all, gt_all, prob_all = torch.zeros((1, )), torch.zeros((1, )), torch.zeros((1, args.num_class))
            embeddings = torch.zeros((1, 768))
            # embeddings = torch.zeros((1, 512))
            threshold = 0
            for counter, sample in enumerate(train_loader):
                x_batch = sample['img'].cuda()
                y_batch = sample['cls_label'].cuda()
                batch_names = sample['img_name']

                # biomedclip
                # image_features, text_features, logit_scale = model_biomedclip(x_batch, texts)
                # probs = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                # plip
                inputs = plip_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch

                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
                outputs = model_plip.forward(**inputs)
                # this is the image-text similarity score
                logits_per_image = outputs.logits_per_image
                probs_plip = logits_per_image.softmax(dim=1)

                # average
                # probs = (probs+probs_plip)/2
                probs = probs_plip

                logits_hard = torch.argmax(probs, dim=1)
                pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
                gt_all = torch.cat((gt_all, y_batch.cpu()), dim=0)
                prob_all = torch.cat((prob_all, probs.cpu()), dim=0)
                names += batch_names
                embeddings = torch.cat((embeddings, outputs.vision_model_output.pooler_output.cpu()), dim=0)
                # embeddings = torch.cat((embeddings, image_features.cpu()), dim=0)
            if j==0:
                probs_n = prob_all.unsqueeze(2)
                pred_n = pred_all.unsqueeze(1)
            else:
                probs_n = torch.cat([probs_n, prob_all.unsqueeze(2)], dim=2)
                pred_n = torch.cat([pred_n, pred_all.unsqueeze(1)], dim=1)
    # print(probs_n.shape)
    # print(pred/gt, (pred/gt).mean())

    pred_all, gt_all, probs_n, pred_n = pred_all[1:], gt_all[1:], probs_n[1:], pred_n[1:]
    embeddings = embeddings[1:]
    names = np.array(names)

    # Here use entropy equals to zero
    pred_n_onehot = F.one_hot(pred_n.long(), num_classes=args.num_class)
    # print(pred_n_onehot.shape)

    # here use uncertainty to select x% most reliable samples
    pred_n_prob = pred_n_onehot.float().mean(1)
    pred_entropy = -torch.sum(torch.log(pred_n_prob+1e-6)*pred_n_prob, dim=1)
    # print(pred_entropy.shape, pred_entropy[:20])
    idx_un = pred_entropy.sort(descending=True)[1].cpu().numpy()
    return pred_entropy.detach().clone().cpu().numpy()

if __name__ == "__main__":
    seed_torch(0)
    args = get_arguments()
    torch.cuda.set_device(args.gpu[0])

    # dataset
    train_files = get_files('/home/ubuntu/data/lanfz/datasets/LC25000/train.csv')
    # train_files = get_files('/home/ubuntu/data/lanfz/datasets/RINGS/train-100-patch/')
    np.random.shuffle(train_files)
    print('train size:', len(train_files))
    train_dataset = Tumor_dataset(args, files=train_files)
    # train_dataset = Tumor_dataset_val(args, files=train_files[int(0.4*len(train_files)):])
    train_dataset_eval = Tumor_dataset_val_cls(args, files=train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False)
    train_eval_loader = get_loader(args, train_dataset_eval, shuffle=False)

    # biomedCLIP
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    plip = CLIPModel.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")
    plip_processor = CLIPProcessor.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")

    # plip = CLIPModel.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/clip-vitb")
    # plip_processor = CLIPProcessor.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/clip-vitb")

    model = model.cuda()
    model.eval()
    plip = plip.cuda()
    plip.eval()

    t1 = time.time()

    text_prompt = ["An H&E image of benign lung tissue", "An H&E image of lung adenocarcinoma", "An H&E image of lung squamous cell carcinoma",\
                   "An H&E image of benign colonic tissue", "An H&E image of colon adenocarcinoma"]
    texts = tokenizer(text_prompt).cuda()

    # zero_shot
    y_pred, y_true, names, embeddings, y_prob = zero_shot_inference(args, train_eval_loader, model, plip, plip_processor)
    cal_acc(y_pred, y_true)
    print('----pred distribution----')
    print(np.bincount(y_pred), np.bincount(y_true))

    # multiple_inference
    pred_entropy = MVC(args, train_loader, model, plip, plip_processor, n_times=10)
    idx_un = np.argsort(pred_entropy)[:int(0.3*pred_entropy.shape[0])]
    y_pred, y_true, names, embeddings, y_prob = y_pred[idx_un], y_true[idx_un], np.array(names)[idx_un], embeddings[idx_un], y_prob[idx_un]
    # cal_acc(y_pred, y_true)
    # cls_recall(args, y_pred, y_true)
    # print('----pred distribution----')
    # print(np.bincount(y_pred), np.bincount(y_true), len(names))

    idx, reorder = Prompt_feature_consensus(args, embeddings.numpy(), y_pred)
    print('reorder matching:', metrics.f1_score(y_pred, reorder, average='macro'))

    y_pred, y_true, names, y_prob = y_pred[idx], y_true[idx], np.array(names)[idx], y_prob[idx]
    # cls_recall(args, y_pred, y_true)
    # cal_acc(y_pred, y_true)
    # print('----pred distribution----')
    # print(np.bincount(y_pred), np.bincount(y_true))

    # write csv
    data_df = pd.DataFrame()
    data_df['image_path'] = names
    data_df['pseudo_label'] = y_pred
    data_df['true_label'] = y_true
    data_df.to_csv('pseudo_csv/LC_pseudo_labels.csv', index=False)
    t2 = time.time()
    print(f"Time: {t2-t1}")
