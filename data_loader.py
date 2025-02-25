import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
import matplotlib.pyplot as plt
import seaborn as sns

class GeneExpressionDataset(Dataset):
    def __init__(self, adata, mask_ratio=0.15):
        self.data = torch.tensor(adata.X, dtype=torch.float32)
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx].clone()
        mask = torch.rand(x.shape) < self.mask_ratio
        x[mask] = 0  # Masking genes
        return x, self.data[idx]  # (Masked input, Original target)

def load_data(proj_root):
    # Load gene expression matrix
    expr_matrix = pd.read_csv(proj_root+"expression_matrix.csv", index_col=0)
    metadata = pd.read_csv("metadata.csv", index_col=0)
    gene_names = pd.read_csv("gene_names.csv", header=None)[1].tolist()
    print("data loaded")


    # Convert data to AnnData format
    adata = sc.AnnData(expr_matrix.T)  # Cells as rows, genes as columns
    adata.obs = metadata  # Assign metadata
    adata.var_names = gene_names

    # Normalize Data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=200)
    adata = adata[:, adata.var.highly_variable]

    return adata, gene_names


