import os
import torch
import torch.utils.data

import pickle
import numpy as np
from skimage.morphology import square, dilation, erosion
import time
from torchvision import transforms as trans
from PIL import Image


def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            distance = np.sqrt(float(i ** 2 + j ** 2))
            if (r+i) >= 0 and (r+i) < height and (c+j) >= 0 and (
                    c + j) < width:
                if 'Solid' == mode and distance <= radius:
                    indices.append([r + i, c + j, k])

    return indices


def _getSparsePose(peaks, height, width,
                   channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0 != len(p):
            r = p[0][1]
            c = p[0][0]
            ind = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)

    shape = [height, width, channel]
    return indices, shape


def _oneDimSparsePose(indices, shape):
    ind_onedim = []
    for ind in indices:
        idx = ind[0] * shape[2] * shape[1] + ind[1] * shape[2] + ind[2]
        ind_onedim.append(idx)
    shape = np.prod(shape)
    return ind_onedim, shape


def _sparse2dense(indices, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r, c, k] = 1
    return dense


def _getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7],
               [7, 8], [2, 9], [9, 10], [10, 11], [2, 12],
               [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [2, 17], [2, 18],
               [9, 12], [12, 6], [9, 3], [17, 18]]
    indices = []
    for limb in limbSeq:
        p0 = peaks[limb[0] - 1]
        p1 = peaks[limb[1] - 1]
        if 0 != len(p0) and 0 != len(p1):
            r0 = p0[0][1]
            c0 = p0[0][0]
            r1 = p1[0][1]
            c1 = p1[0][0]
            ind = _getSparseKeypoint(
                r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            ind = _getSparseKeypoint(
                r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)

            distance = np.sqrt((r0 - r1) ** 2 + (c0 - c1) ** 2)
            sampleN = int(distance / radius)
            if sampleN > 1:
                for i in range(1, sampleN):
                    r = r0 + (r1 - r0) * i / sampleN
                    c = c0 + (c1 - c0) * i / sampleN
                    ind = _getSparseKeypoint(
                        r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)

    shape = [height, width, 1]

    dense = np.squeeze(_sparse2dense(indices, shape))
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense


def _get_valid_peaks(all_peaks, subsets):
    try:
        subsets = subsets.tolist()
        valid_idx = -1
        valid_score = -1
        for i, subset in enumerate(subsets):
            score = subset[-2]
            if score > valid_score:
                valid_idx = i
                valid_score = score
        if valid_idx >= 0:
            return all_peaks
        else:
            return None
    except:
        return None


def _format_data(img_dir, pairs, i, all_peaks_dic,
                 subsets_dic, transforms=None):

    # Read the filename:
    img_path_0 = os.path.join(img_dir, pairs[i][0])
    img_path_1 = os.path.join(img_dir, pairs[i][1])
    image_raw_0 = Image.open(img_path_0)
    image_raw_1 = Image.open(img_path_1)
    width, height, _= np.array(image_raw_0).shape

    # --- Pose 16x8 & Pose coodinate (for 128x64(Solid) 128x64(Gaussian)) --- #
    if (all_peaks_dic is not None) \
            and (pairs[i][0] in all_peaks_dic) \
            and (pairs[i][1] in all_peaks_dic):
        ## Pose 1
        peaks = _get_valid_peaks(
            all_peaks_dic[pairs[i][1]], subsets_dic[pairs[i][1]])
        indices_r4_1, shape = _getSparsePose(
            peaks, height, width, 18, radius=4, mode='Solid')
        indices_r4_1, shape_1 = _oneDimSparsePose(indices_r4_1, shape)
        pose_mask_r4_1 = _getPoseMask(
            peaks, height, width, radius=4, mode='Solid')

    else:
        print("None, Again -_- ...")
        return None

    mask_1 = np.reshape(pose_mask_r4_1, (height, width, 1))
    mask_1 = mask_1.astype('float32')

    indices_r4_1 = np.array(indices_r4_1).astype(np.int64).flatten().tolist()
    indices_r4_1_dense = np.zeros((shape_1))
    indices_r4_1_dense[indices_r4_1] = 1
    indices_r4_1 = np.reshape(indices_r4_1_dense, (height, width, 18))
    pose_1 = indices_r4_1.astype('float32')

    image_0 = transforms(image_raw_0)
    image_1 = transforms(image_raw_1)
    pose_1 = pose_1 * 2 - 1

    mask_1 = torch.from_numpy(np.transpose(mask_1, (2, 0, 1)))
    pose_1 = torch.from_numpy(np.transpose(pose_1, (2, 0, 1)))


    """
    image_0:  torch.Size([3, 256, 256])
    image_1:  torch.Size([3, 256, 256])
    pose_1:  torch.Size([18, 256, 256])
    mask_1:  torch.Size([1, 256, 256])
    """
    return_dic = {
        'image_0': image_0,
        'image_1': image_1,
        'pose_1': pose_1,
        'mask_1': mask_1,
        'img_path_0': img_path_0,
        'img_path_1': img_path_1
    }

    return return_dic


class PoseDataset(torch.utils.data.Dataset):
    """Pose dataset."""

    def __init__(self, opt, img_dir, p_pairs_file,
                 pose_peak_file, pose_sub_file, transforms=None):
        t = time.time()
        self.opt = opt
        self.img_dir = img_dir
        self.transforms = transforms
        with open(p_pairs_file, 'rb') as f:
            self.p_pairs = pickle.load(f)

        self.length = len(self.p_pairs)
        print('Sum of pairs: ', self.length)

        self.all_peaks_dic = None
        self.subsets_dic = None

        with open(pose_peak_file, 'rb') as f:
            self.all_peaks_dic = pickle.load(f, encoding='latin1')
        with open(pose_sub_file, 'rb') as f:
            self.subsets_dic = pickle.load(f, encoding='latin1')

        if self.opt.control_data_num is not None:
            self.p_pairs = self.p_pairs[0:self.opt.control_data_num]
            self.length = len(self.p_pairs)
        print('Experiment Use Pairs: %d' % self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        while True:
            example = _format_data(self.img_dir,
                                   self.p_pairs,
                                   index,
                                   self.all_peaks_dic,
                                   self.subsets_dic,
                                   transforms=self.transforms)
            if example is not None:
                return example
            index = (index + 1) % self.length


def get_loader(opt):
    split_name = 'train' if opt.isTrain else 'test'

    img_dir = os.path.join(opt.dataset_dir, 'filted_up_' + split_name)
    others_dir = os.path.join(opt.dataset_dir, 'new_DF_data')

    data_transforms = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5])
    ])

    pose_dataset = PoseDataset(
        opt=opt,
        img_dir=img_dir,
        p_pairs_file=os.path.join(others_dir, split_name + '_n_pairs.p'),
        pose_peak_file=os.path.join(others_dir, split_name + '_all_peaks_dic_DeepFashion.p'),
        pose_sub_file=os.path.join(others_dir, split_name + '_subsets_dic_DeepFashion.p'),
        transforms=data_transforms
    )

    pose_loader = torch.utils.data.DataLoader(
        pose_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    return pose_loader


if __name__ == '__main__':
    from config import configs
    from utils import *
    import matplotlib.pyplot as plt

    opt = configs()
    opt.control_data_num = None
    opt.isTrain = True

    data = get_loader(opt)

    count = 0
    for step, data_dic in enumerate(data):
        # data_ = open('./data_dic.pkl', 'wb')
        # pickle.dump(data_dic, data_)

        source_img = data_dic['image_0']
        target_img = data_dic['image_1']

        plt.imshow(tensor2im(source_img))
        plt.show()
        plt.imshow(tensor2im(target_img))
        plt.show()

        break
