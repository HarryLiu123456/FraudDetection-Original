
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

 
# # **Data Preparation**

 
# **Data Overview**


import numpy as np
import pandas as pd

 
# **Data Description**


transaction_df = pd.read_csv('./input/train_transaction.csv')
identity_df = pd.read_csv('./input/train_identity.csv')
test_transaction = pd.read_csv('./input/test_transaction.csv')
test_identity = pd.read_csv('./input/test_identity.csv')


id_cols = ['card1','card2','card3','card4','card5','card6','ProductCD','addr1','addr2','P_emaildomain','R_emaildomain']
cat_cols = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']
train_data_ratio = 0.8


n_train = int(transaction_df.shape[0]*train_data_ratio)
test_ids = transaction_df.TransactionID.values[n_train:]


get_fraud_frac = lambda series: 100 * sum(series)/len(series)
print("Percent fraud for train transactions: {}".format(get_fraud_frac(transaction_df.isFraud[:n_train])))
print("Percent fraud for test transactions: {}".format(get_fraud_frac(transaction_df.isFraud[n_train:])))
print("Percent fraud for all transactions: {}".format(get_fraud_frac(transaction_df.isFraud)))


with open('./working/test.csv', 'w') as f:
    f.writelines(map(lambda x: str(x) + "\n", test_ids))


non_feature_cols = ['isFraud', 'TransactionDT'] + id_cols
print(non_feature_cols)


feature_cols = [col for col in transaction_df.columns if col not in non_feature_cols]
print(feature_cols)


features = pd.get_dummies(transaction_df[feature_cols], columns=cat_cols).fillna(0)
features['TransactionAmt'] = features['TransactionAmt'].apply(np.log10)


features.shape


print(list(features.columns))


features.to_csv('./working/features.csv', index=False, header=False)


transaction_df[['TransactionID', 'isFraud']].to_csv('./working/tags.csv', index=False)


edge_types = id_cols + list(identity_df.columns)
print(edge_types)


all_id_cols = ['TransactionID'] + id_cols
full_identity_df = transaction_df[all_id_cols].merge(identity_df, on='TransactionID', how='left')
print(all_id_cols)
print(full_identity_df.head(5))


edges = {}
for etype in edge_types:
    edgelist = full_identity_df[['TransactionID', etype]].dropna()
    edgelist.to_csv('./working/relation_{}_edgelist.csv'.format(etype), index=False, header=True)
    edges[etype] = edgelist

print(edges)

 
# # **Modeling**

 
# **installs**


import os
import sys
import glob

os.environ['DGLBACKEND'] = 'pytorch'

import torch as th
import dgl
import numpy as np

 
# *estimator_fns*


import os
import argparse
import logging
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dir', type=str, default='./working')
    dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    parser.add_argument('--model-dir', type=str, default='./working'+dt)
    parser.add_argument('--output-dir', type=str, default='./working')
    parser.add_argument('--nodes', type=str, default='features.csv')
    parser.add_argument('--target-ntype', type=str, default='TransactionID')
    parser.add_argument('--edges', type=str, default='./working/relation*')
    parser.add_argument('--labels', type=str, default='tags.csv')
    parser.add_argument('--new-accounts', type=str, default='test.csv')
    parser.add_argument('--compute-metrics', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='compute evaluation metrics after training')
    parser.add_argument('--threshold', type=float, default=0, help='threshold for making predictions, default : argmax')
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--n-hidden', type=int, default=32, help='number of hidden units') 
    parser.add_argument('--n-layers', type=int, default=6, help='number of hidden layers')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight for L2 loss')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability, for gat only features')
    parser.add_argument('--embedding-size', type=int, default=360, help="embedding size for node embedding")
    return parser.parse_known_args()[0]
 
# *graph_utils*


import os
import re
import dgl
import numpy as np
import torch as th
import logging
#from data import parse_edgelist, get_features, read_edges, read_masked_nodes
import pandas as pd

def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger

logging = get_logger(__name__)


def get_edgelists(edgelist_expression, directory):
    if "," in edgelist_expression:
        return edgelist_expression.split(",")
    files = os.listdir(directory)
    compiled_expression = re.compile(edgelist_expression)
    return [filename for filename in files if compiled_expression.match(filename)]


def construct_graph(training_dir, edges, nodes, target_node_type):

    print("Getting relation graphs from the following edge lists : {} ".format(edges))
    edgelists, id_to_node = {}, {}
    for i, edge in enumerate(edges):
        edgelist, rev_edgelist, id_to_node, src, dst = parse_edgelist(os.path.join(training_dir, edge), id_to_node, header=True)
        if src == target_node_type:
            src = 'target'
        if dst == target_node_type:
            dst = 'target'

        if src == 'target' and dst == 'target':
            print("Will add self loop for target later......")
        else:
            edgelists[(src, src + '<>' + dst, dst)] = edgelist
            edgelists[(dst, dst + '<>' + src, src)] = rev_edgelist
            print("Read edges for {} from edgelist: {}".format(src + '<' + dst + '>', os.path.join(training_dir, edge)))

    # get features for target nodes
    features, new_nodes = get_features(id_to_node[target_node_type], os.path.join(training_dir, nodes))
    print("Read in features for target nodes")

    # add self relation
    edgelists[('target', 'self_relation', 'target')] = [(t, t) for t in id_to_node[target_node_type].values()]

    g = dgl.heterograph(edgelists)
    print(
        "Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
            g.ntypes, g.canonical_etypes))
    print("Number of nodes of type target : {}".format(g.number_of_nodes('target')))

    g.nodes['target'].data['features'] = th.from_numpy(features)



    target_id_to_node = id_to_node[target_node_type]
    id_to_node['target'] = target_id_to_node

    del id_to_node[target_node_type]

    return g, features, target_id_to_node, id_to_node


# *data*


import numpy as np
import pandas as pd

# def string_to_float(x):
#     if(isinstance(x, str)):
#         if(x=='False'):
#             return 0.0
#         elif(x=='True'):
#             return 1.0
#     else:
#         return x.float()

def get_features(id_to_node, node_features):
    """
    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param node_features: path to file containing node features
    :return: (np.ndarray, list) node feature matrix in order and new nodes not yet in the graph
    """
    indices, features, new_nodes = [], [], []
    max_node = max(id_to_node.values())

    is_1st_line = True
    with open(node_features, "r") as fh:
        for line in fh:
            node_feats = line.strip().split(",")
            node_id = node_feats[0]
            feats = np.array(list(map(float, node_feats[1:])))
            features.append(feats)
            if node_id not in id_to_node:
                max_node += 1
                id_to_node[node_id] = max_node
                new_nodes.append(max_node)

            indices.append(id_to_node[node_id])

    features = np.array(features).astype('float32')
    features = features[np.argsort(indices), :]
    return features, new_nodes



def get_labels(id_to_node, n_nodes, target_node_type, labels_path, masked_nodes_path, additional_mask_rate=0):
    """
    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param n_nodes: number of user nodes in the graph
    :param target_node_type: column name for target node type
    :param labels_path: filepath containing labelled nodes
    :param masked_nodes_path: filepath containing list of nodes to be masked
    :param additional_mask_rate: additional_mask_rate: float for additional masking of nodes with labels during training
    :return: (list, list) train and test mask array
    """
    node_to_id = {v: k for k, v in id_to_node.items()}
    user_to_label = pd.read_csv(labels_path).set_index(target_node_type)
    labels = user_to_label.loc[map(int, pd.Series(node_to_id)[np.arange(n_nodes)].values)].values.flatten()
    masked_nodes = read_masked_nodes(masked_nodes_path)
    train_mask, test_mask = _get_mask(id_to_node, node_to_id, n_nodes, masked_nodes,
                                      additional_mask_rate=additional_mask_rate)
    return labels, train_mask, test_mask


def read_masked_nodes(masked_nodes_path):
    """
    Returns a list of nodes extracted from the path passed in
    :param masked_nodes_path: filepath containing list of nodes to be masked i.e test users
    :return: list
    """
    with open(masked_nodes_path, "r") as fh:
        masked_nodes = [line.strip() for line in fh]
    return masked_nodes


def _get_mask(id_to_node, node_to_id, num_nodes, masked_nodes, additional_mask_rate):
    """
    Returns the train and test mask arrays
    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param node_to_id: dictionary mapping dgl node idx to node names(id)
    :param num_nodes: number of user/account nodes in the graph
    :param masked_nodes: list of nodes to be masked during training, nodes without labels
    :param additional_mask_rate: float for additional masking of nodes with labels during training
    :return: (list, list) train and test mask array
    """
    train_mask = np.ones(num_nodes)
    test_mask = np.zeros(num_nodes)
    for node_id in masked_nodes:
        train_mask[id_to_node[node_id]] = 0
        test_mask[id_to_node[node_id]] = 1
    if additional_mask_rate and additional_mask_rate < 1:
        unmasked = np.array([idx for idx in range(num_nodes) if node_to_id[idx] not in masked_nodes])
        yet_unmasked = np.random.permutation(unmasked)[:int(additional_mask_rate*num_nodes)]
        train_mask[yet_unmasked] = 0
    return train_mask, test_mask



def _get_node_idx(id_to_node, node_type, node_id, ptr):
    if node_type in id_to_node:
        if node_id in id_to_node[node_type]:
            node_idx = id_to_node[node_type][node_id]
        else:
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1
    else:
        id_to_node[node_type] = {}
        id_to_node[node_type][node_id] = ptr
        node_idx = ptr
        ptr += 1

    return node_idx, id_to_node, ptr



def parse_edgelist(edges, id_to_node, header=False, source_type='user', sink_type='user'):
    """
    Parse an edgelist path file and return the edges as a list of tuple
    :param edges: path to comma separated file containing bipartite edges with header for edgetype
    :param id_to_node: dictionary containing mapping for node names(id) to dgl node indices
    :param header: boolean whether or not the file has a header row
    :param source_type: type of the source node in the edge. defaults to 'user' if no header
    :param sink_type: type of the sink node in the edge. defaults to 'user' if no header.
    :return: (list, dict) a list containing edges of a single relationship type as tuples and updated id_to_node dict.
    """
    edge_list = []
    rev_edge_list = []
    source_pointer, sink_pointer = 0, 0
    with open(edges, "r") as fh:
        for i, line in enumerate(fh):
            source, sink = line.strip().split(",")
            if i == 0:
                if header:
                    source_type, sink_type = source, sink
                if source_type in id_to_node:
                    source_pointer = max(id_to_node[source_type].values()) + 1
                if sink_type in id_to_node:
                    sink_pointer = max(id_to_node[sink_type].values()) + 1
                continue

            source_node, id_to_node, source_pointer = _get_node_idx(id_to_node, source_type, source, source_pointer)
            if source_type == sink_type:
                sink_node, id_to_node, source_pointer = _get_node_idx(id_to_node, sink_type, sink, source_pointer)
            else:
                sink_node, id_to_node, sink_pointer = _get_node_idx(id_to_node, sink_type, sink, sink_pointer)

            edge_list.append((source_node, sink_node))
            rev_edge_list.append((sink_node, source_node))

    return edge_list, rev_edge_list, id_to_node, source_type, sink_type



def read_edges(edges, nodes=None):
    """
    Read edges and node features
    :param edges: path to comma separated file containing all edges
    :param nodes: path to comma separated file containing all nodes + features
    :return: (list, list, list, dict) sources, sinks, features and id_to_node dictionary containing mappings
    from node names(id) to dgl node indices
    """
    node_pointer = 0
    id_to_node = {}
    features = []
    sources, sinks = [], []
    if nodes is not None:
        with open(nodes, "r") as fh:
            for line in fh:
                node_feats = line.strip().split(",")
                node_id = node_feats[0]
                if node_id not in id_to_node:
                    id_to_node[node_id] = node_pointer
                    node_pointer += 1
                    if len(node_feats) > 1:
                        feats = np.array(list(map(float, node_feats[1:])))
                        features.append(feats)
        with open(edges, "r") as fh:
            for line in fh:
                source, sink = line.strip().split(",")
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])
    else:
        with open(edges, "r") as fh:
            for line in fh:
                source, sink = line.strip().split(",")
                if source not in id_to_node:
                    id_to_node[source] = node_pointer
                    node_pointer += 1
                if sink not in id_to_node:
                    id_to_node[sink] = node_pointer
                    node_pointer += 1
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])

    return sources, sinks, features, id_to_node

 
# *utils*


import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import networkx as nx
import matplotlib.pyplot as plt


def get_metrics(pred, pred_proba, labels, mask, out_dir):
    labels, mask = labels, mask
    labels, pred, pred_proba = labels[np.where(mask)], pred[np.where(mask)], pred_proba[np.where(mask)]
    acc = ((pred == labels)).sum() / mask.sum()
    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    precision = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2*(precision*recall)/(precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = pd.DataFrame(np.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])
    ap = average_precision_score(labels, pred_proba)
    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)
    save_roc_curve(fpr, tpr, roc_auc, os.path.join(out_dir, "roc_curve.png"))
    save_pr_curve(prc, rec, pr_auc, ap, os.path.join(out_dir, "pr_curve.png"))

    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix


def save_roc_curve(fpr, tpr, roc_auc, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC curve')
    plt.legend(loc="lower right")
    f.savefig(location)


def save_pr_curve(fpr, tpr, pr_auc, ap, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Model PR curve: AP={0:0.2f}'.format(ap))
    plt.legend(loc="lower right")
    f.savefig(location)


def save_graph_drawing(g, location):
    plt.figure(figsize=(12, 8))
    node_colors = {node: 0.0 if 'user' in node else 0.5 for node in g.nodes()}
    nx.draw(g, node_size=10000, pos=nx.spring_layout(g), with_labels=True, font_size=14,
            node_color=list(node_colors.values()), font_color='white')
    plt.savefig(location, bbox_inches='tight')

 
# *pytorch_model*


import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import dgl.function as fn


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                # Save it in graph for message passing
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                # Specify per-relation message passing functions: (message_func, reduce_func).
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.nodes[ntype].data}


class HeteroRGCN(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, embedding_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
                      for ntype, num_nodes in ntype_dict.items() if ntype != 'target'}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(embedding_size, hidden_size, etypes))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))

        # output layer
        self.layers.append(nn.Linear(hidden_size, out_size))

    def forward(self, g, features):
        # get embeddings for all node types. for user node type, use passed in user features
        h_dict = {ntype: emb for ntype, emb in self.embed.items()}
        # feat_para = torch.tensor(features)
        h_dict['target'] = features

        # pass through all layers
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g, h_dict)

        # get user logits
        return self.layers[-1](h_dict['target'])

 
# ***train***


os.environ['DGLBACKEND'] = 'pytorch'

import torch as th
import dgl
print('DLG version: {}'.format(dgl.__version__))
import numpy as np
import pandas as pd
import time
import pickle
import copy


from sklearn.metrics import confusion_matrix


def initial_record():
    if os.path.exists('./working/results.txt'):
        os.remove('./working/results.txt')
    with open('./working/results.txt','w') as f:    
        f.write("Epoch,Time(s),Loss,F1\n")   


def normalize(feature_matrix):
    mean = th.mean(feature_matrix, axis=0)
    stdev = th.sqrt(th.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0])
    return mean, stdev, (feature_matrix - mean) / stdev


def train_fg(model, optim, loss, features, labels, train_g, test_g, test_mask,
             device, n_epochs, thresh, compute_metrics=True):
    """
    A full graph verison of RGCN training
    """

    duration = []
    best_loss = 1
    for epoch in range(n_epochs):
        tic = time.time()
        loss_val = 0.

        pred = model(train_g, features.to(device))

        l = loss(pred, labels)

        optim.zero_grad()
        l.backward()
        optim.step()

        loss_val += l

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, device)
        print("Epoch {:05d}, Time(s) {:.4f}, Loss {:.4f}, F1 {:.4f} ".format(
                epoch, np.mean(duration), loss_val, metric))
        
        epoch_result = "{:05d},{:.4f},{:.4f},{:.4f}\n".format(epoch, np.mean(duration), loss_val, metric)
        with open('./working/results.txt','a+') as f:    
            f.write(epoch_result)  

        if loss_val < best_loss:
            best_loss = loss_val
            best_model = copy.deepcopy(model)


    class_preds, pred_proba = get_model_class_predictions(best_model,
                                                          test_g,
                                                          features,
                                                          labels,
                                                          device,
                                                          threshold=thresh)

    if compute_metrics:
        acc, f1, p, r, roc, pr, ap, cm = get_metrics(class_preds, pred_proba, labels.numpy(), test_mask.numpy(), './working')
        print("Metrics")
        print("""Confusion Matrix:
                                {}
                                f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}, pr: {:.4f}, ap: {:.4f}
                             """.format(cm, f1, p, r, acc, roc, pr, ap))

    return best_model, class_preds, pred_proba


def get_f1_score(y_true, y_pred):
    """
    Only works for binary case.
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]
    :param y_true: A list of labels in 0 or 1: 1 * N
    :param y_pred: A list of labels in 0 or 1: 1 * N
    :return:
    """
    # print(y_true, y_pred)

    cf_m = confusion_matrix(y_true, y_pred)
    # print(cf_m)

    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)

    return precision, recall, f1



def evaluate(model, g, features, labels, device):
    "Compute the F1 value in a binary classification case"

    preds = model(g, features.to(device))
    preds = th.argmax(preds, axis=1).numpy()
    precision, recall, f1 = get_f1_score(labels, preds)

    return f1



def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    unnormalized_preds = model(g, features.to(device))
    pred_proba = th.softmax(unnormalized_preds, dim=-1)
    if not threshold:
        return unnormalized_preds.argmax(axis=1).detach().numpy(), pred_proba[:,1].detach().numpy()
    return np.where(pred_proba.detach().numpy() > threshold, 1, 0), pred_proba[:,1].detach().numpy()


def save_model(g, model, model_dir, id_to_node, mean, stdev):

    # Save Pytorch model's parameters to model.pth
    th.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    # Save graph's structure information to metadata.pkl for inference codes to initialize RGCN model.
    etype_list = g.canonical_etypes
    ntype_cnt = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({'etypes': etype_list,
                     'ntype_cnt': ntype_cnt,
                     'feat_mean': mean,
                     'feat_std': stdev}, f)

    # Save original IDs to Node_ids, and trained embedding for non-target node type
    # Covert id_to_node into pandas dataframes
    for ntype, mapping in id_to_node.items():

        # ignore target node
        if ntype == 'target':
            continue

        # retrieve old and node id list
        old_id_list, node_id_list = [], []
        for old_id, node_id in mapping.items():
            old_id_list.append(old_id)
            node_id_list.append(node_id)

        # retrieve embeddings of a node type
        node_feats = model.embed[ntype].detach().numpy()

        # get the number of nodes and the dimension of features
        num_nodes = node_feats.shape[0]
        num_feats = node_feats.shape[1]

        # create id dataframe
        node_ids_df = pd.DataFrame({'~label': [ntype] * num_nodes})
        node_ids_df['~id_tmp'] = old_id_list
        node_ids_df['~id'] = node_ids_df['~label'] + '-' + node_ids_df['~id_tmp']
        node_ids_df['node_id'] = node_id_list

        # create feature dataframe columns
        cols = {'val' + str(i + 1) + ':Double': node_feats[:, i] for i in range(num_feats)}
        node_feats_df = pd.DataFrame(cols)

        # merge id with feature, where feature_df use index
        node_id_feats_df = node_ids_df.merge(node_feats_df, left_on='node_id', right_on=node_feats_df.index)
        # drop the id_tmp and node_id columns to follow the Grelim format requirements
        node_id_feats_df = node_id_feats_df.drop(['~id_tmp', 'node_id'], axis=1)

        # dump the embeddings to files
        node_id_feats_df.to_csv(os.path.join(model_dir, ntype + '.csv'),
                                index=False, header=True, encoding='utf-8')


def get_model(ntype_dict, etypes, hyperparams, in_feats, n_classes, device):

    model = HeteroRGCN(ntype_dict, etypes, in_feats, hyperparams['n_hidden'], n_classes, hyperparams['n_layers'], in_feats)
    model = model.to(device)

    return model

############################################################################################################
# **Load data**


file_list = glob.glob('./working/*edgelist.csv')

edges = ",".join(map(lambda x: x.split("/")[-1], [file for file in file_list if "relation" in file]))

 
# **Generate graph**


print('numpy version:{} PyTorch version:{} DGL version:{}'.format(np.__version__,
                                                                    th.__version__,
                                                                    dgl.__version__))

args = parse_args()
print(args)


args.edges = edges

args.edges = get_edgelists('relation*', args.training_dir)

g, features, target_id_to_node, id_to_node = construct_graph(args.training_dir,
                                                                args.edges,
                                                                args.nodes,
                                                                args.target_ntype)

mean, stdev, features = normalize(th.from_numpy(features))

print('feature mean shape:{}, std shape:{}'.format(mean.shape, stdev.shape))


g.nodes['target'].data['features'] = features

print("Getting labels")
n_nodes = g.number_of_nodes('target')

labels, _, test_mask = get_labels(target_id_to_node,
                                            n_nodes,
                                            args.target_ntype,
                                            os.path.join(args.training_dir, args.labels),
                                            os.path.join(args.training_dir, args.new_accounts))
print("Got labels")

labels = th.from_numpy(labels).float()
test_mask = th.from_numpy(test_mask).float()

n_nodes = th.sum(th.tensor([g.number_of_nodes(n_type) for n_type in g.ntypes]))
n_edges = th.sum(th.tensor([g.number_of_edges(e_type) for e_type in g.etypes]))

print("""----Data statistics------'
            #Nodes: {}
            #Edges: {}
            #Features Shape: {}
            #Labeled Test samples: {}""".format(n_nodes,
                                                    n_edges,
                                                    features.shape,
                                                    test_mask.sum()))

 
# **Training**


if args.num_gpus:
    cuda = True
    device = th.device('cuda:0')
else:
    cuda = False
    device = th.device('cpu')


print("Initializing Model")
in_feats = features.shape[1]
n_classes = 2


ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}


model = get_model(ntype_dict, g.etypes, vars(args), in_feats, n_classes, device)
print("Initialized Model")


features = features.to(device)


labels = labels.long().to(device)
test_mask = test_mask.to(device)


loss = th.nn.CrossEntropyLoss()


# print(model)
optim = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)



print("Starting Model training")


initial_record()


model, class_preds, pred_proba = train_fg(model, optim, loss, features, labels, g, g,
                                            test_mask, device, args.n_epochs,
                                            args.threshold,  args.compute_metrics)


