import ra_preproc as ra_preproc
import mo_sql_parsing as msp
import json
import pickle
from tqdm import tqdm
from apted import APTED, Config
import numpy as np
import os
from anytree import PreOrderIter
import itertools
import multiprocessing
from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument("dataset-source", help="Path to .json file containing input data", type=str)
# parser.add_argument("dataset-target", help="Path to .json file meant to store output data", type=str)

class CustomConfig(Config):
    def __init__(self):
        super().__init__()
        # all_names={'union', 'Orderby_desc', 'gt', 'lte', 'Table', 'Project', 'Selection', 'eq', 'sum', 'Subquery', 'Or', 'keep', 'Val_list', 'distinct', 'like', 'in', 'Value', 'lt', 'literal', 'intersect', 'Limit', 'except', 'neq', 'And', 'gte', 'max', 'Groupby', 'avg', 'count', 'min', 'nin', 'Orderby_asc', 'Product'}

        self.agg_grp=['max','min','avg','count','sum']
        self.order_grp=['Orderby_desc','Orderby_asc']
        self.boolean_grp=['Or','And']
        self.set_grp=['union','intersect','except']
        self.leaf_grp=['Val_list','Value','literal','Table']
        self.sim_grp=['like','in','nin']
        self.comp_grp=['gt','lte','eq','lt','gte','neq']
        self.groups = [
            self.agg_grp, self.order_grp, self.boolean_grp, self.set_grp,
            self.leaf_grp, self.sim_grp, self.comp_grp,
        ]
        self.reverse_dict = {}
        for i, group in enumerate(self.groups):
            for name in group:
                self.reverse_dict[name] = i

    def in_same_grp(self, name1, name2):
        try:
            return self.reverse_dict[name1] == self.reverse_dict[name2]
        except KeyError as k:
            return False

    def rename(self, node1, node2):
        if self.in_same_grp(node1.name, node2.name):
            return 1 if node1.name != node2.name else 0
        else:
            return 2 if node1.name != node2.name else 0

    def children(self, node):
        return [x for x in node.children]

config = CustomConfig()

def parse_data(data):
    result = []
    for i, item in enumerate(tqdm(data, total=len(data))):
        try:
            # tree_dict = msp.parse(item['query'])
            tree_dict = msp.parse(item)
            tree_obj = ra_preproc.ast_to_ra(tree_dict)
            size = len(list(PreOrderIter(tree_obj)))
            result.append([tree_obj, size, tree_dict, item])
        except Exception as e:
            print(f"Index {i}, error: {e}")
            continue
    return result


def compute_ted(x, y):
    try:
        d_i, d_j = parse_data([x, y])
    except Exception as e:
        return 10000, 10000
    tree_obj_i, size_i, _, _ = d_i
    if tree_obj_i is None:
        tgt_sizes[i] = 1e-6
        return 10000, 10000
    tree_obj_j, size_j, _, _ = d_j
    if tree_obj_j is None:
        return 10000, 10000
    try:
        dist = APTED(tree_obj_i, tree_obj_j, config).compute_edit_distance()
    except Exception as e:
        return 10000, 10000
    assert dist >= 0
    normalized = dist / max(size_j, size_i)
    return dist, normalized
    

# # SOURCE_JSON = '/mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/train_spider.json'
# TARGET_JSON = '/mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/dev.json'

# # SOURCE_JSON = '/mnt/infonas/data/shashankshet/picard/shashank_exp/ted/train-data.json'
# SOURCE_JSON = '/mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/dev.json'
# source_data_orig = json.load(open(SOURCE_JSON))
# target_data_orig = json.load(open(TARGET_JSON))

# source_data = parse_data(source_data_orig)
# target_data = parse_data(target_data_orig)

# src_len, tgt_len = len(source_data), len(target_data)
# print(src_len, tgt_len)
# src_sizes, tgt_sizes = np.zeros(src_len), np.zeros(tgt_len)
# dist_matrix_train = np.ones([src_len, src_len]) * 1e9
# dist_matrix_test  = np.ones([tgt_len, src_len]) * 1e9
# dist_matrix_train_norm = np.ones([src_len, src_len]) * 1e9
# dist_matrix_test_norm  = np.ones([tgt_len, src_len]) * 1e9


# train_idxs = [(i,j) for i in range(src_len) for j in range(src_len)]
# # test_idxs = [(i,j) for i in range(src_len) for j in range(tgt_len)]
# test_idxs = []
# for i in range(tgt_len):
#     for j in range(src_len):
#         test_idxs.append((i,j))

# def process_test_i_j(idx):
#     i, j = idx
#     tree_obj_i, size_i, _, _ = target_data[i]
#     if tree_obj_i is None:
#         tgt_sizes[i] = 1e-6
#         return
#     else:
#         tgt_sizes[i] = size_i
#     tree_obj_j, size_j, _, _ = source_data[j]
#     if tree_obj_j is None:
#         src_sizes[j] = 1e-6
#         return
#     else:
#         src_sizes[j] = size_j
#     dist = APTED(tree_obj_i, tree_obj_j, config).compute_edit_distance()
#     assert dist >= 0
#     normalized = dist / max(size_j, size_i)
#     return dist, normalized

# def process_train_i_j(idx):
#     i, j = idx
#     tree_obj_i, size_i, _, _ = source_data[i]
#     if tree_obj_i is None:
#         src_sizes[i] = 1e-6
#         return
#     else:
#         src_sizes[i] = size_i
#     tree_obj_j, size_j, _, _ = source_data[j]
#     if tree_obj_j is None:
#         src_sizes[j] = 1e-6
#         return
#     else:
#         src_sizes[j] = size_j
#     dist = APTED(tree_obj_i, tree_obj_j, config).compute_edit_distance()
#     assert dist >= 0
#     normalized = dist / max(size_j, size_i)
#     return dist, normalized

# num_procs = min(multiprocessing.cpu_count(), 16)
# print(num_procs)

# with multiprocessing.Pool(num_procs) as p:
#     res = list(tqdm(p.imap(process_train_i_j, train_idxs), total=len(train_idxs)))

# for res_i, idx in enumerate(train_idxs):
#     i, j = idx
#     if i==j:
#         continue
#     dist, normalized = res[res_i]
#     dist_matrix_train[i, j] = dist
#     dist_matrix_train_norm[i, j] = normalized

# # cbr_train_indices = dist_matrix_train[i, j].argmin(axis=-1)
# np.save("cbr_sim_matrix_test2", dist_matrix_train)
# np.save("cbr_sim_matrix_test2_norm", dist_matrix_train_norm)

# exit(0)

# print("Train set cases fixed")

# with multiprocessing.Pool(num_procs) as p:
#     res = list(tqdm(p.imap(process_test_i_j, test_idxs), total=len(test_idxs)))

# for res_i, idx in enumerate(test_idxs):
#     i, j = idx
#     dist, normalized = res[res_i]
#     dist_matrix_test[i, j] = dist
#     dist_matrix_test_norm[i, j] = normalized

# # cbr_test_indices = dist_matrix_test[i, j].argmin(axis=-1)
# np.save("cbr_sim_matrix_test", dist_matrix_test)
# np.save("cbr_sim_matrix_test_norm", dist_matrix_test_norm)

# print("Test set cases fixed")


# if (node1.name in self.agg_grp and node2.name in self.agg_grp) or \
# (node1.name in self.order_grp and node2.name in self.order_grp) or \
# (node1.name in self.boolean_grp and node2.name in self.boolean_grp) or \
# (node1.name in self.set_grp and node2.name in self.set_grp) or \
# (node1.name in self.leaf_grp and node2.name in self.leaf_grp) or \
# (node1.name in self.sim_grp and node2.name in self.sim_grp) or \
# (node1.name in self.comp_grp and node2.name in self.comp_grp):
