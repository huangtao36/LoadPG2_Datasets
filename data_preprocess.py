import os
from utils import mkdirs
import pickle
from easydict import EasyDict as edit

opt = edit({
    'dataset_dir': './DeepFashion_FromPG2'  # dataroot 
})

split_name = 'train'  # 'test'

out_dir = os.path.join(opt.dataset_dir, 'DF_'+ split_name +'_data')
pose_peak_path = os.path.join(opt.dataset_dir, 'PoseFiltered', 'all_peaks_dic_DeepFashion.p')
pose_sub_path = os.path.join(opt.dataset_dir, 'PoseFiltered',  'subsets_dic_DeepFashion.p')

save_root = os.path.join(opt.dataset_dir, 'new_DF_'+ split_name +'_data')
mkdirs(save_root)

with open(pose_peak_path, 'rb') as f:
    all_peaks_dic = pickle.load(f, encoding='latin1')

def _get_all_p_pairs(out_dir, split_name='train'):
    assert split_name in {'train', 'test'}

    p_pairs_path = os.path.join(
        out_dir, 'p_pairs_' + split_name + '.p')

    p_pairs = None
    if os.path.exists(p_pairs_path):
        with open(p_pairs_path, 'rb') as f:
            p_pairs = pickle.load(f)

    print('_get_train_all_pn_pairs finish ...')
    print('p_pairs length:%d' % len(p_pairs))

    return p_pairs

p_pairs = _get_all_p_pairs(out_dir, split_name=split_name)

# get new p_pairs
new_p_pairs = []
for i in range(len(p_pairs)):
    if (p_pairs[i][0] in all_peaks_dic) and (p_pairs[i][1] in all_peaks_dic):
        new_p_pairs.append([p_pairs[i][0], p_pairs[i][1]])

# save to .p
pkl_file = os.path.join(
    opt.dataset_dir, 'new_DF_data', split_name+'_n_pairs.p')
file_ = open(pkl_file, 'wb')
pickle.dump(new_p_pairs, file_)


# # for all_peaks_dic
# print("----------------all_peaks_dic----------------")
# count = 0
# new_all_peaks_dic = {}
# for i in range(len(new_p_pairs)):
#     new_all_peaks_dic[new_p_pairs[i][0]] = all_peaks_dic[new_p_pairs[i][0]]
#     new_all_peaks_dic[new_p_pairs[i][1]] = all_peaks_dic[new_p_pairs[i][1]]
#     count += 1
#     if count % 10000 == 0:
#         print(count)
#
# # save args to .p
# pkl_file = os.path.join(
#     save_root, split_name + '_all_peaks_dic_DeepFashion.p')
# file_ = open(pkl_file, 'wb')
# pickle.dump(new_all_peaks_dic, file_)
#

# -----------------------------------------------------------------------------
# # for subsets_dic
# with open(pose_sub_path, 'rb') as f:
#     subsets_dic = pickle.load(f, encoding='latin1')
#
# print("----------------subsets_dic----------------")
# count = 0
# new_subsets_dic = {}
# for i in range(len(new_p_pairs)):
#     new_subsets_dic[new_p_pairs[i][0]] = subsets_dic[new_p_pairs[i][0]]
#     new_subsets_dic[new_p_pairs[i][1]] = subsets_dic[new_p_pairs[i][1]]
#     count += 1
#     if count % 10000 == 0:
#         print(count)
#
# # save args to .p
# pkl_file = os.path.join(
#     save_root, split_name + '_subsets_dic_DeepFashion.p')
# file_ = open(pkl_file, 'wb')
# pickle.dump(new_subsets_dic, file_)
#