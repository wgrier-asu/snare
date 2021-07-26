import numpy as np
import collections
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from models.single_cls import SingleClassifier


class Rotator(SingleClassifier):

    def __init__(self, cfg, train_ds, val_ds):
        super().__init__(cfg, train_ds, val_ds)

    def build_model(self):
        # image encoder
        self.img_fc = nn.Sequential(
            nn.Identity()
        )

        # language encoder
        self.lang_fc = nn.Sequential(
            nn.Identity()
        )

        # finetuning layers for classification
        self.cls_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim+self.lang_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        # load pre-trained classifier (gets overrided if loading pre-trained rotator)
        # Note: gets overrided if loading pre-trained rotator
        model_path = self.cfg['train']['rotator']['pretrained_cls']
        self.load_state_dict(torch.load(model_path)['state_dict'])
        print(f"Loaded: {model_path}")

        # state estimation layers
        self.state_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8)
        )

        # action layers
        self.action_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim+self.lang_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8)
        )

        # load pre-trained rotator
        if self.cfg['train']['pretrained_model']:
            model_path = self.cfg['train']['pretrained_model']
            self.load_state_dict(torch.load(model_path)['state_dict'])
            print(f"Loaded: {model_path}")


    def forward(self, batch, teacher_force=True, init_view_force=None, final_view_force=None):
        (img1_n_feats, img2_n_feats), lang_feats, ans, (key1, key2), annotation, is_visual = batch

        # rotate to and estimate current view
        next_views_out = self.estimate_state(img1_n_feats, img2_n_feats, lang_feats,
                                             ans, teacher_force, init_view_force, final_view_force)

        # output variables from state estimation
        bs = img1_n_feats.shape[0]

        img1_n_feats = next_views_out['img1_n_feats']
        img2_n_feats = next_views_out['img2_n_feats']
        lang_feats = next_views_out['lang_feats']

        init_views1 = next_views_out['init_views1']
        init_views2 = next_views_out['init_views2']

        est_init_views1 = next_views_out['est_init_views1']
        est_init_views2 = next_views_out['est_init_views2']

        loss = next_views_out['loss']

        # choose features of ramdomly sampling viewpoints
        img1_chosen_feats, img2_chosen_feats = self.choose_feats_from_random_views(bs, img1_n_feats, img2_n_feats,
                                                                                   init_views1, init_views2)

        # classifier probablities chosen features
        img1_chosen_prob = self.cls_fc(torch.cat([img1_chosen_feats, lang_feats], dim=-1))
        img2_chosen_prob = self.cls_fc(torch.cat([img2_chosen_feats, lang_feats], dim=-1))

        # classifier loss
        raw_probs = torch.cat([img1_chosen_prob, img2_chosen_prob], dim=-1)
        probs = F.softmax(raw_probs, dim=-1)
        bs = lang_feats.shape[0]
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_feats.device) * 2

        test_mode = (ans[0] == -1)
        if not test_mode:
        # classifier loss
            cls_labels = F.one_hot(ans)
            cls_loss_weight = self.cfg['train']['loss']['cls_weight']
            loss += (self.smoothed_cross_entropy(raw_probs, cls_labels)) * cls_loss_weight

            # state estimation accuracy
            est_view1_corrects = int(torch.count_nonzero(est_init_views1 == init_views1))
            est_view2_corrects = int(torch.count_nonzero(est_init_views2 == init_views2))
            total_correct_estimations = est_view1_corrects + est_view2_corrects

            # state estimation error
            est_err = torch.cat([self.modulo_views(init_views1 - est_init_views1).abs().float(),
                                 self.modulo_views(init_views2 - est_init_views2).abs().float()])
            est_err = est_err.mean()

            return {
                'probs': probs,
                'action_loss': loss,
                'labels': cls_labels,
                'is_visual': is_visual,
                'num_steps': num_steps,

                'total_correct_estimations': total_correct_estimations,
                'est_error': est_err,
                'est_init_views1': est_init_views1,
                'est_init_views2': est_init_views2,
            }
        else:
            return {
                'probs': probs,
                'num_steps': num_steps,
            }

    def estimate_state(self, img1_n_feats, img2_n_feats, lang_feats, ans, teacher_force,
                       init_view_force, final_view_force):
        # to device
        img1_n_feats = img1_n_feats.to(device=self.device).float()
        img2_n_feats = img2_n_feats.to(device=self.device).float()
        lang_feats = lang_feats.to(device=self.device).float()

        all_probs = []
        bs = img1_n_feats.shape[0]
        num_views = img1_n_feats.shape[1]

        # lang encoding
        lang_feats = self.lang_fc(lang_feats)

        # normalize
        if self.cfg['train']['normalize_feats']:
            img1_n_feats /= img1_n_feats.norm(dim=-1, keepdim=True)
            img2_n_feats /= img2_n_feats.norm(dim=-1, keepdim=True)
            lang_feats /= lang_feats.norm(dim=-1, keepdim=True)

        # compute single_cls probs for 8 view pairs
        for v in range(num_views):
            # aggregate
            img1_feats = img1_n_feats[:, v]
            img2_feats = img2_n_feats[:, v]

            # img1 prob
            img1_feats = self.img_fc(img1_feats)
            img1_prob = self.cls_fc(torch.cat([img1_feats, lang_feats], dim=-1))

            # img2 prob
            img2_feats = self.img_fc(img2_feats)
            img2_prob = self.cls_fc(torch.cat([img2_feats, lang_feats], dim=-1))

            # cat probs
            view_probs = torch.cat([img1_prob, img2_prob], dim=-1)
            all_probs.append(view_probs)

        all_probs = torch.stack(all_probs, dim=1)
        all_probs = F.softmax(all_probs, dim=2)

        # best views with highest classifier probs
        best_views1 = all_probs[:, :, 0].argmax(-1)
        best_views2 = all_probs[:, :, 1].argmax(-1)

        # worst views with lowest classifier probs
        worst_views1 = all_probs[:, :, 0].argmin(-1)
        worst_views2 = all_probs[:, :, 0].argmin(-1)

        # Initialize with worst views
        if init_view_force == 'adv':
            init_views1 = worst_views1
            init_views2 = worst_views2
        else:
            # initialize with random views
            if init_view_force is None:
                init_views1 = torch.randint(num_views, (bs,)).cuda()
                init_views2 = torch.randint(num_views, (bs,)).cuda()
            else:
                init_views1 = torch.ones((bs,)).int().cuda() * init_view_force
                init_views2 = torch.ones((bs,)).int().cuda() * init_view_force

        # init features
        img1_init_feats = torch.stack([img1_n_feats[i, init_views1[i], :] for i in range(bs)])
        img2_init_feats = torch.stack([img2_n_feats[i, init_views2[i], :] for i in range(bs)])

        # state estimator
        est_init_views_logits1 = self.state_fc(img1_init_feats)
        est_init_views_logits2 = self.state_fc(img2_init_feats)

        gt_init_views1 = F.one_hot(init_views1.to(torch.int64), num_classes=8)
        gt_init_views2 = F.one_hot(init_views2.to(torch.int64), num_classes=8)

        # state estimation loss
        est_loss_weight = self.cfg['train']['loss']['est_weight']
        loss = ((self.smoothed_cross_entropy(est_init_views_logits1, gt_init_views1) +
                 self.smoothed_cross_entropy(est_init_views_logits2, gt_init_views2)) / 2) * est_loss_weight

        est_init_views1 = F.softmax(est_init_views_logits1, dim=-1).argmax(-1)
        est_init_views2 = F.softmax(est_init_views_logits2, dim=-1).argmax(-1)

        return {
            'best_views1': best_views1,
            'best_views2': best_views2,
            'img1_n_feats': img1_n_feats,
            'img2_n_feats': img2_n_feats,
            'lang_feats': lang_feats,
            'loss': loss,
            'init_views1': init_views1,
            'init_views2': init_views2,
            'est_init_views1': est_init_views1,
            'est_init_views2': est_init_views2,
        }

    def modulo_views(self, views):
        bs = views.shape[0]
        modulo_views = torch.zeros_like(views)
        for b in range(bs):
            view = views[b]

            if view < 4 and view >= -4:
                modulo_views[b] = view
            elif view >= 4:
                modulo_views[b] = -4 + (view % 4)
            elif view < -4:
                modulo_views[b] = 4 - (abs(view) % 4)
        return modulo_views

    def choose_feats_from_random_views(self, bs, img1_n_feats, img2_n_feats, init_views1, init_views2):
        rand_next_views = torch.randint(8, (2, bs))
        img1_chosen_feats = torch.stack([img1_n_feats[i, [init_views1[i], rand_next_views[0, i]], :].max(dim=-2)[0]
                                       for i in range(bs)])
        img2_chosen_feats = torch.stack([img2_n_feats[i, [init_views2[i], rand_next_views[1, i]], :].max(dim=-2)[0]
                                       for i in range(bs)])
        return img1_chosen_feats, img2_chosen_feats

    def compute_metrics(self, labels, loss, probs, visual, num_steps, total_correct_estimations):
        batch_size = probs.shape[0]
        val_total, val_correct, val_pl_correct = 0, 0, 0.
        visual_total, visual_correct, pl_visual_correct = 0, 0, 0.
        nonvis_total, nonvis_correct, pl_nonvis_correct = 0, 0, 0.
        for b in range(batch_size):
            correct = self.check_correct(b, labels, probs)

            if correct:
                val_correct += 1
                val_pl_correct += 1. / num_steps[b]
            val_total += 1

            if bool(visual[b]):
                if correct:
                    visual_correct += 1
                    pl_visual_correct += 1. / num_steps[b]
                visual_total += 1
            else:
                if correct:
                    nonvis_correct += 1
                    pl_nonvis_correct += 1. / num_steps[b]
                nonvis_total += 1

        correct_ests = total_correct_estimations
        total_rots = 2 * batch_size

        val_acc = float(val_correct) / val_total
        val_pl_acc = float(val_pl_correct) / val_total
        val_visual_acc = float(visual_correct) / visual_total
        val_pl_visual_acc = float(pl_visual_correct) / visual_total
        val_nonvis_acc = float(nonvis_correct) / nonvis_total
        val_pl_nonvis_acc = float(pl_nonvis_correct) / nonvis_total
        val_est_err = float(correct_ests) / total_rots

        return dict(
                val_loss=loss,
                val_acc=val_acc,
                val_pl_acc=val_pl_acc,
                val_correct=val_correct,
                val_pl_correct=val_pl_correct,
                val_total=val_total,
                val_visual_acc=val_visual_acc,
                val_pl_visual_acc=val_pl_visual_acc,
                val_visual_correct=visual_correct,
                val_pl_visual_correct=pl_visual_correct,
                val_visual_total=visual_total,
                val_nonvis_acc=val_nonvis_acc,
                val_pl_nonvis_acc=val_pl_nonvis_acc,
                val_nonvis_correct=nonvis_correct,
                val_pl_nonvis_correct=pl_nonvis_correct,
                val_nonvis_total=nonvis_total,
                val_est_err=val_est_err
            )

    def training_step(self, batch, batch_idx):
        out = self.forward(batch, teacher_force=self.cfg['train']['rotator']['teacher_force'])

        if self.log_data:
            wandb.log({
                'tr/loss': out['action_loss'],
            })

        return dict(
            loss=out['action_loss']
        )

    def validation_step(self, batch, batch_idx):
        all_view_results = {}
        views = list(range(self.num_views))
        for view in views:
            # view selection
            if self.cfg['val']['adversarial_init_view']:
                out = self.forward(batch, teacher_force=False, init_view_force='adv')
            else:
                out = self.forward(batch, teacher_force=False, init_view_force=view)

            # losses
            losses = self._criterion(out)

            loss = losses['loss']
            probs = out['probs']
            labels = out['labels']
            visual = out['is_visual']
            num_steps = out['num_steps']
            total_correct_estimations = out['total_correct_estimations']

            metrics = self.compute_metrics(labels, loss, probs, visual, num_steps, total_correct_estimations)
            all_view_results[view] = metrics

        mean_val_loss = np.mean([m['val_loss'].detach().cpu().float() for m in all_view_results.values()])
        mean_val_acc = np.mean([m['val_acc'] for m in all_view_results.values()])

        return dict(
            val_loss=mean_val_loss,
            val_acc=mean_val_acc,

            all_view_results=all_view_results,
        )

    def validation_epoch_end(self, all_outputs, mode='vl'):
        n_view_res = {}
        views = list(range(self.num_views))

        for view in views:

            view_res = {
                'val_loss': 0.0,

                'val_correct': 0,
                'val_pl_correct': 0,
                'val_total': 0,

                'val_visual_correct': 0,
                'val_pl_visual_correct': 0,
                'val_visual_total': 0,

                'val_nonvis_correct': 0,
                'val_pl_nonvis_correct': 0,
                'val_nonvis_total': 0,

                'val_est_err': 0.0,
            }

            for output in all_outputs:
                metrics = output['all_view_results'][view]

                view_res['val_loss'] += metrics['val_loss'].item()

                view_res['val_correct'] += metrics['val_correct']
                view_res['val_pl_correct'] += int(metrics['val_pl_correct'])
                view_res['val_total'] += metrics['val_total']

                view_res['val_visual_correct'] += metrics['val_visual_correct']
                view_res['val_pl_visual_correct'] += int(metrics['val_pl_visual_correct'])
                view_res['val_visual_total'] += metrics['val_visual_total']

                view_res['val_nonvis_correct'] += metrics['val_nonvis_correct']
                view_res['val_pl_nonvis_correct'] += int(metrics['val_pl_nonvis_correct'])
                view_res['val_nonvis_total'] += metrics['val_nonvis_total']

                view_res['val_est_err'] += int(metrics['val_est_err'])

            view_res['val_loss'] = float(view_res['val_loss']) / len(all_outputs)

            view_res['val_acc'] = float(view_res['val_correct']) / view_res['val_total']
            view_res['val_pl_acc'] = float(view_res['val_pl_correct']) / view_res['val_total']

            view_res['val_visual_acc'] = float(view_res['val_visual_correct']) / view_res['val_visual_total']
            view_res['val_pl_visual_acc'] = float(view_res['val_pl_visual_correct']) / view_res['val_visual_total']

            view_res['val_nonvis_acc'] = float(view_res['val_nonvis_correct']) / view_res['val_nonvis_total']
            view_res['val_pl_nonvis_acc'] = float(view_res['val_pl_nonvis_correct']) / view_res['val_nonvis_total']

            view_res['val_est_err'] = float(view_res['val_est_err']) / len(all_outputs)

            n_view_res[view] = view_res

        mean_val_loss = np.mean([r['val_loss'] for r in n_view_res.values()])

        val_acc = sum([r['val_correct'] for r in n_view_res.values()]) / float(sum([r['val_total'] for r in n_view_res.values()]))
        val_visual_acc = sum([r['val_visual_correct'] for r in n_view_res.values()]) / float(sum([r['val_visual_total'] for r in n_view_res.values()]))
        val_nonvis_acc = sum([r['val_nonvis_correct'] for r in n_view_res.values()]) / float(sum([r['val_nonvis_total'] for r in n_view_res.values()]))

        val_pl_acc = sum([r['val_pl_correct'] for r in n_view_res.values()]) / float(sum([r['val_total'] for r in n_view_res.values()]))
        val_pl_visual_acc = sum([r['val_pl_visual_correct'] for r in n_view_res.values()]) / float(sum([r['val_visual_total'] for r in n_view_res.values()]))
        val_pl_nonvis_acc = sum([r['val_pl_nonvis_correct'] for r in n_view_res.values()]) / float(sum([r['val_nonvis_total'] for r in n_view_res.values()]))

        val_est_err = np.mean([r['val_est_err'] for r in n_view_res.values()])

        res = {
            f'{mode}/loss': mean_val_loss,
            f'{mode}/acc': val_acc,
            f'{mode}/acc_visual': val_visual_acc,
            f'{mode}/acc_nonvis': val_nonvis_acc,
            f'{mode}/pl_acc': val_pl_acc,
            f'{mode}/pl_acc_visual': val_pl_visual_acc,
            f'{mode}/pl_acc_nonvis': val_pl_nonvis_acc,
            f'{mode}/est_err': val_est_err,
            f'{mode}/all_view_res': n_view_res,
        }

        if mode == 'test':
            self.best_test_res = dict(res)
        else:
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_res = dict(res)

        dump_res = self.best_test_res if mode == 'test' else self.best_val_res

        # print best result
        print("\nBest-----:")
        best_loss = dump_res[f'{mode}/loss']
        best_acc = dump_res[f'{mode}/acc']
        best_acc_visual = dump_res[f'{mode}/acc_visual']
        best_acc_nonvis = dump_res[f'{mode}/acc_nonvis']
        best_pl_acc = dump_res[f'{mode}/pl_acc']
        best_pl_acc_visual = dump_res[f'{mode}/pl_acc_visual']
        best_pl_acc_nonvis = dump_res[f'{mode}/pl_acc_nonvis']
        best_est_err = dump_res[f'{mode}/est_err']

        seed = self.cfg['train']['random_seed']
        json_file = os.path.join(self.save_path, f'{mode}-results-{seed}.json')
        with open(json_file, 'w') as f:
            json.dump(dump_res, f, sort_keys=True, indent=4)

        print(f'Curr Acc: {res[f"{mode}/acc"]:0.5f} ({res[f"{mode}/pl_acc"]:0.5f}) | Visual {res[f"{mode}/acc_visual"]:0.5f} ({res[f"{mode}/pl_acc_visual"]:0.5f}) | Nonvis: {res[f"{mode}/acc_nonvis"]:0.5f} ({res[f"{mode}/pl_acc_nonvis"]:0.5f}) | Avg. Est Err: {res[f"{mode}/est_err"]:0.5f} | Val Loss: {res[f"{mode}/loss"]:0.8f} ')
        print(f'Best Acc: {best_acc:0.5f} ({best_pl_acc:0.5f}) | Visual {best_acc_visual:0.5f} ({best_pl_acc_visual:0.5f}) | Nonvis: {best_acc_nonvis:0.5f} ({best_pl_acc_nonvis:0.5f}) | Avg. Est Err: {best_est_err:0.5f} | Val Loss: {best_loss:0.8f} ')
        print("------------")

        if self.log_data:
            wandb.log(res)
        return dict(
            val_loss=mean_val_loss,
            val_acc=val_acc,
            val_visual_acc=val_visual_acc,
            val_nonvis_acc=val_nonvis_acc,
            val_pl_acc=val_pl_acc,
            val_pl_visual_acc=val_pl_visual_acc,
            val_pl_nonvis_acc=val_pl_nonvis_acc,
        )

