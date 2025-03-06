import torch.nn as nn
from data_loader import *
from transformers import BertModel, BertConfig

class GeneBERT(nn.Module):
    def __init__(self, input_dim, hidden_dim=768):
        super(GeneBERT, self).__init__()
        config = BertConfig(
            vocab_size=input_dim,  # Using the number of genes as vocab size
            hidden_size=hidden_dim,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072
        )
        #self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(hidden_dim, input_dim)  # Output layer for gene prediction
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x_embed = self.embedding(x)
        x_embed = x_embed.unsqueeze(1)
        output = self.bert(inputs_embeds=x_embed, output_attentions=True)  # Treat input as embeddings
        attention_weights = output.attentions[-1]  # Last layer's attention
        return self.fc(output.last_hidden_state[:, 0, :]), attention_weights  # Predict expression


#var
proj_root = "/Users/chen/Library/CloudStorage/GoogleDrive-schen601@usc.edu/My Drive/transportor/"

# data load
adata, gene_names = load_data(proj_root)

# Model initialization
device = torch.device("mps")
model = GeneBERT(input_dim=adata.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Regression loss for gene expression prediction



num_epochs = 5
# Create dataset
dataset = GeneExpressionDataset(adata)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        masked_input, target = batch
        masked_input, target = masked_input.to(device), target.to(device)

        optimizer.zero_grad()
        predictions = model(masked_input)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


# Extract attention weights
attention_weights = model.bert.encoder.layer[-1].attention.self.get_attn_probs()

# Select top transporter genes (e.g., GLUT1, ABCB1, AQP4)
transporter_genes = ["Slc7a5", "Slc2a1","Slc22a8"]
transporter_indices = [adata.var_names.get_loc(g) for g in transporter_genes]

# Extract and visualize attention for transporters
transporter_attention = attention_weights[:, transporter_indices, :].mean(dim=0).cpu().detach().numpy()

sns.heatmap(transporter_attention, cmap="coolwarm", xticklabels=gene_names, yticklabels=transporter_genes)
plt.title("Gene Attention Map for Transporters")
plt.show()

