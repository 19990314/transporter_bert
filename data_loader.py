import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
import matplotlib.pyplot as plt
import seaborn as sns

# Load gene expression matrix
expr_matrix = pd.read_csv("expression_matrix.csv", index_col=0)
metadata = pd.read_csv("metadata.csv", index_col=0)
gene_names = pd.read_csv("gene_names.csv", header=None)[0].tolist()

# Convert data to AnnData format
adata = sc.AnnData(expr_matrix.T)  # Cells as rows, genes as columns
adata.obs = metadata  # Assign metadata
adata.var_names = gene_names

# Normalize Data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select highly variable genes
sc.pp.highly_variable_genes(adata, flavor="seurat_v5", n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]