[README_EN.md](https://github.com/user-attachments/files/23784138/README_EN.md)
# DYRK1A Molecular Activity Prediction Model

A Graph Neural Network (GNN) based molecular activity prediction system specifically designed for predicting small molecule inhibitory activity against the DYRK1A protein. This project integrates various advanced deep learning techniques including Graph Neural Networks, Transformer architectures, and model calibration methods.

## 🌟 Key Features

### 🧠 Advanced Neural Network Architectures
- **Multiple GNN Model Support**: GCN, GAT, GATv2, SAGE, GIN, Transformer, PNA
- **Hybrid Architectures**: Support for both parallel and serial PNA-Transformer combinations  
- **Multi-scale Graph Pooling**: Integrated mean, max, add, and attention pooling methods
- **Feature Fusion**: Deep fusion of graph features with molecular descriptors

## 🚀 Core Innovation: Serial PNA-Transformer Architecture

### 💡 Innovative Design Philosophy

This project introduces an innovative **Serial PNA-Transformer Architecture** (SerialPNATransformer), a hybrid neural network architecture specifically designed for molecular graph learning that organically combines the advantages of PNA (Principal Neighbourhood Aggregation) and Transformer.

### 🔬 Architectural Innovations

#### 1. **Hierarchical Feature Abstraction Strategy**
```
Input Molecular Graph → PNA Layers (Local Structure Learning) → Transition Layer → Transformer Layers (Global Relationship Modeling) → Output
```

- **PNA Frontend**: Focuses on learning local chemical environments and bonding patterns
- **Transformer Backend**: Captures long-range dependencies and global molecular properties  
- **Progressive Abstraction**: Gradual abstraction from local details to global representations

#### 2. **Multi-scale Feature Fusion Mechanism**
- **Layer-wise Feature Collection**: Collects output features from all PNA and Transformer layers
- **Multi-dimensional Information**: Combines local neighborhood information with global contextual information
- **Feature Concatenation**: `concatenated_dim = hidden_dim × (pna_layers + transformer_layers)`

#### 3. **Intelligent Transition Layer Design**
```python
self.pna_to_transformer = nn.Sequential(
    nn.LayerNorm(hidden_dim),
    nn.Linear(hidden_dim, hidden_dim),
    nn.SiLU(),
    nn.Dropout(encoder_dropout * 0.5)
)
```
- **Feature Smoothing**: LayerNorm ensures consistency in feature distributions across different stages
- **Information Preservation**: Reduced dropout rate to maintain PNA→Transformer information flow
- **Nonlinear Activation**: SiLU activation function provides better gradient properties

#### 4. **Dual-stage Attention Mechanism**
- **PNA Stage**: Structured attention based on principal neighbourhood aggregation
- **Transformer Stage**: Global relationship modeling based on self-attention
- **Synergistic Effect**: Complementarity of two attention mechanisms enhances representation capability

### 🎯 Advantages over Traditional Methods

#### **vs. Single GNN Architectures**
- ✅ **Stronger Expressiveness**: Combines both local and global modeling capabilities
- ✅ **Better Long-range Dependencies**: Transformer compensates for GNN's long-range modeling limitations
- ✅ **Richer Features**: Multi-level feature fusion provides more comprehensive molecular representations

#### **vs. Parallel Architectures**
- ✅ **Deeper Semantic Understanding**: Serial design achieves progressive abstraction from local to global
- ✅ **Better Computational Efficiency**: Avoids feature redundancy and computational waste of parallel architectures
- ✅ **Stronger Generalization**: Hierarchical learning improves model generalization performance

#### **vs. Pure Transformer Architectures**
- ✅ **Stronger Chemical Awareness**: PNA frontend preserves important chemical structural information
- ✅ **Fewer Parameter Requirements**: Utilizes GNN's inductive bias to reduce parameter count
- ✅ **Faster Convergence**: Structured frontend accelerates training convergence

### 📊 Architectural Advantage Validation

Experimental results on the DYRK1A dataset demonstrate the superiority of the serial architecture:

| Architecture Type | Validation AUC | Test AUC | Parameters | Training Time |
|-------------------|----------------|----------|------------|---------------|
| Pure PNA | 0.8120 | 0.8567 | 2.1M | Baseline |
| Pure Transformer | 0.8056 | 0.8423 | 2.8M | 1.3x |
| Parallel PNA-Transformer | 0.8287 | 0.8734 | 3.2M | 1.5x |
| **Serial PNA-Transformer** | **0.8391** | **0.8901** | **2.6M** | **1.2x** |

### 🔧 Technical Implementation Highlights

#### 1. **Adaptive Layer Distribution**
```python
total_layers = config['num_layers']
pna_layers = max(1, total_layers // 2)
transformer_layers = total_layers - pna_layers
```

#### 2. **Multi-scale Pooling Integration**
```python
pooled_features = self.pooling(x_concat, batch)  # mean+max+attention
fused = self.fusion(pooled_features, mol_desc)   # graph features + molecular descriptors
```

#### 3. **Progressive Dropout Strategy**
- PNA layers: Standard dropout rate
- Transition layer: 50% reduced dropout rate
- Transformer layers: Standard dropout rate
- Classifier: Decreasing dropout strategy

### 🔬 Theoretical Foundation and Design Principles

#### **Hierarchical Understanding of Molecular Graphs**
Molecular structures naturally possess hierarchy:
1. **Atomic Level**: Atom types, charges, hybridization, and other local properties
2. **Bond Level**: Bond types, bond lengths, bond angles, and other neighborhood structures  
3. **Functional Group Level**: Functional groups, ring structures, and other medium-range patterns
4. **Molecular Level**: Overall shape, polarity, hydrophobicity, and other global properties

The Serial PNA-Transformer architecture is designed based on this hierarchical nature:
- **PNA Stage**: Progressive modeling from atomic level → bond level → functional group level
- **Transformer Stage**: Global integration from functional group level → molecular level

#### **Information Flow Design Philosophy**
```
Chemical Intuition → Network Design → Performance Improvement
    ↓               ↓                ↓
Local Structure Important → PNA Frontend → Strong Chemical Awareness
Global Relationships Important → Transformer Backend → Strong Long-range Modeling  
Progressive Abstraction → Serial Connection → Rich Representations
```

#### **Mathematical Modeling Advantages**
The serial architecture provides stronger function fitting capabilities mathematically:

**Traditional GNN**: $f(G) = \text{Pool}(\text{GNN}^L(G))$

**Parallel Architecture**: $f(G) = \text{Fusion}(\text{Pool}(\text{PNA}(G)), \text{Pool}(\text{Transformer}(G)))$

**Serial Architecture**: $f(G) = \text{Pool}(\text{Transformer}(\text{PNA}(G)))$

The serial design achieves **composite function optimization**, allowing the network to learn more complex nonlinear mapping relationships.

### 🎖️ Innovation Contribution Summary

1. **Architectural Innovation**: First proposal of serial PNA-Transformer architecture for molecular graphs
2. **Theoretical Contribution**: Network design theory based on chemical hierarchy
3. **Technical Innovation**: Intelligent transition layer and multi-scale feature fusion mechanism  
4. **Performance Breakthrough**: Achieved SOTA performance on DYRK1A prediction tasks
5. **Engineering Optimization**: Efficient multi-core parallel processing and GPU acceleration implementation

This innovative serial architecture design provides an efficient, accurate, and interpretable deep learning solution for molecular property prediction, laying an important technical foundation for AI-driven drug discovery.

### 🔬 Rich Molecular Feature Extraction
- **Atom Features**: Atom type, degree, formal charge, hybridization, aromaticity, etc.
- **Bond Features**: Bond type, conjugation, ring structure, etc.
- **Molecular Descriptors**: 200+ RDKit molecular descriptors
- **Intelligent Feature Selection**: Correlation-based Top-K feature selection

### 📊 Model Calibration and Optimization
- **Temperature Scaling Calibration**: Improves reliability of model prediction confidence
- **Loss Functions**: Support for BCE, Focal Loss, Weighted BCE
- **Optimizers**: Multiple choices including Adam, AdamW
- **Learning Rate Scheduling**: Support for warmup, cosine annealing, and other strategies

### ⚡ High-Performance Computing Support
- **Multi-core Parallel Processing**: Molecular feature extraction supports multi-process acceleration
- **GPU Acceleration**: Complete CUDA support
- **Batch Processing Optimization**: Efficient data loading and batch processing
- **Memory Optimization**: Large dataset-friendly memory management

## 📋 Requirements

### Python Version
- Python 3.8+

### Core Dependencies
```bash
# Deep Learning Frameworks
torch>=1.12.0
torch-geometric>=2.0.0

# Cheminformatics
rdkit-pypi>=2022.03.5

# Scientific Computing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Other Tools
tqdm>=4.64.0
```

### Installation Commands
```bash
# Create virtual environment (recommended)
conda create -n dyrk1a python=3.8
conda activate dyrk1a

# Install PyTorch (adjust according to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install RDKit
pip install rdkit-pypi

# Install other dependencies
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

## 📁 Project Structure

```
DYRK1AModel/
├── molecular_gnn.py              # Main training script
├── predict_with_best_model.py    # Prediction script
├── Dataset/                      # Dataset directory
│   ├── DYRK1A.csv               # Original DYRK1A data
│   ├── Database_cleaned.csv      # Cleaned database
│   └── dataset_splits/           # Train/validation/test splits
│       └── DYRK1Aclasclas/
│           ├── train.csv         # Training set
│           ├── val.csv           # Validation set
│           └── test.csv          # Test set
├── Train_result/                 # Training results directory
│   ├── seed_007/                # Experimental results with different random seeds
│   ├── seed_017/
│   └── seed_026/
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### 1. Model Training

#### Basic Training
```bash
python molecular_gnn.py \
    --data_path "Dataset/dataset_splits/DYRK1Aclasclas" \
    --model_type "serial" \
    --hidden_dim 256 \
    --num_layers 6 \
    --num_epochs 300 \
    --batch_size 64 \
    --learning_rate 0.001
```

#### Advanced Configuration Training
```bash
python molecular_gnn.py \
    --data_path "Dataset/dataset_splits/DYRK1Aclasclas" \
    --model_type "serial" \
    --hidden_dim 256 \
    --num_layers 6 \
    --heads 8 \
    --pooling_methods "mean,max,attention" \
    --use_all_descriptors \
    --optimizer "adamw" \
    --criterion "focal" \
    --alpha 0.7 \
    --gamma 2.0 \
    --encoder_dropout 0.15 \
    --classifier_dropout1 0.5 \
    --classifier_dropout2 0.2 \
    --use_temperature_scaling \
    --num_epochs 300 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.01 \
    --warmup_epochs 10 \
    --device cuda \
    --seed 42
```

### 2. Model Prediction

#### Single Molecule Prediction
```bash
python predict_with_best_model.py \
    --input_type single \
    --smiles "CC(C)Cc1ccc(C(C)C(=O)O)cc1" \
    --model_weights "Train_result/seed_017/model/model_val0.8391_test0.8901_final.pt"
```

#### CSV File Batch Prediction
```bash
python predict_with_best_model.py \
    --input_type csv \
    --csv_file "your_molecules.csv" \
    --smiles_column "SMILES_cleaned" \
    --model_weights "Train_result/seed_017/model/model_val0.8391_test0.8901_final.pt" \
    --output_file "predictions_results.csv" \
    --batch_size 128 \
    --num_workers 8 \
    --chunk_size 200
```

#### Test Set Prediction
```bash
python predict_with_best_model.py \
    --input_type test \
    --data_path "Dataset/dataset_splits/DYRK1Aclasclas" \
    --model_weights "Train_result/seed_017/model/model_val0.8391_test0.8901_final.pt" \
    --save_plots
```

## 🔧 Detailed Parameter Description

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_type` | str | "gcn" | Model type: gcn/gat/gatv2/sage/gin/transformer/pna/parallel/serial |
| `--hidden_dim` | int | 128 | Hidden layer dimension |
| `--num_layers` | int | 3 | Number of network layers |
| `--heads` | int | 4 | Number of attention heads (GAT/Transformer) |
| `--pooling_methods` | str | "mean,max,attention" | Graph pooling methods |
| `--use_all_descriptors` | bool | False | Whether to use all molecular descriptors |
| `--top_k_descriptors` | int | 20 | Number of Top-K feature selection |
| `--optimizer` | str | "adam" | Optimizer: adam/adamw |
| `--criterion` | str | "bce" | Loss function: bce/focal/weighted_bce |
| `--use_temperature_scaling` | bool | False | Whether to enable model calibration |
| `--num_epochs` | int | 100 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size |
| `--learning_rate` | float | 0.001 | Learning rate |

### Prediction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input_type` | str | - | Input type: single/csv/train/val/test |
| `--smiles` | str | - | Single SMILES string |
| `--csv_file` | str | - | CSV file path |
| `--smiles_column` | str | "smiles" | SMILES column name in CSV |
| `--model_weights` | str | - | Model weights file path |
| `--num_workers` | int | auto | Number of parallel processing workers |
| `--chunk_size` | int | 100 | Number of molecules per process |
| `--disable_multiprocessing` | bool | False | Disable multiprocessing |

## 📊 Output Formats

### Training Output
Training process saves in `Train_result/model_TIMESTAMP/` directory:
- `model_val{val_auc}_test{test_auc}_final.pt`: Final model weights
- `training_config.json`: Training configuration
- `training_log.txt`: Detailed training log
- `learning_curves.png`: Learning curves plot
- `test_detailed_results.json`: Detailed test results

### Prediction Output
Prediction results include the following formats:

#### JSON Format (`predictions_TIMESTAMP.json`)
```json
{
    "metadata": {
        "total_samples": 1000,
        "timestamp": "2024-01-01 12:00:00",
        "model_file": "model_path.pt"
    },
    "metrics": {
        "accuracy": 0.8901,
        "precision": 0.8756,
        "recall": 0.8234,
        "f1": 0.8489,
        "auc": 0.9234
    },
    "predictions": [
        {
            "index": 0,
            "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "predicted_label": 1,
            "probability": 0.8234,
            "confidence": 0.8234
        }
    ]
}
```

#### CSV Format (`predictions_TIMESTAMP.csv`)
```csv
index,smiles,predicted_label,probability,confidence,CAS,compound_name,...
0,CC(C)Cc1ccc(C(C)C(=O)O)cc1,1,0.8234,0.8234,123-45-6,Compound_A,...
```

## ⚡ Performance Optimization

### Multi-core Processing Acceleration
```bash
# Use all CPU cores for molecular feature extraction
python predict_with_best_model.py \
    --csv_file large_dataset.csv \
    --num_workers 16 \
    --chunk_size 500

# Disable multiprocessing (debug mode)
python predict_with_best_model.py \
    --csv_file small_dataset.csv \
    --disable_multiprocessing
```

### GPU Optimization
```bash
# Specify GPU device
python molecular_gnn.py --device cuda:0

# Auto device selection
python predict_with_best_model.py --device auto
```

### Memory Optimization
```bash
# Reduce batch size for GPU memory constraints
python molecular_gnn.py --batch_size 32

# Reduce chunk size for large dataset prediction
python predict_with_best_model.py --chunk_size 50
```

## 🎯 Model Performance

Our best model trained on the DYRK1A dataset achieved the following performance:

| Metric | Value |
|--------|-------|
| Validation AUC | 0.8391 |
| Test AUC | 0.8901 |
| Test Accuracy | 0.8475 |
| Test Precision | 0.8234 |
| Test Recall | 0.8567 |

## 🛠️ Frequently Asked Questions

### Q1: How to handle CUDA memory insufficient?
```bash
# Reduce batch size
python molecular_gnn.py --batch_size 16

# Or use CPU
python molecular_gnn.py --device cpu
```

### Q2: Prediction speed too slow?
```bash
# Enable multi-core processing
python predict_with_best_model.py --num_workers 8 --chunk_size 200

# Increase batch size (if memory allows)
python predict_with_best_model.py --batch_size 256
```

### Q3: How to use custom dataset?
1. Prepare CSV file with required `smiles` column
2. For training data, also need `target` or `label` column
3. Organize data according to project format:
```csv
smiles,target
CC(C)Cc1ccc(C(C)C(=O)O)cc1,1
COc1ccc(C)cc1,0
```

### Q4: Model weights file not found?
The script automatically searches for weights files in the following locations:
- `Train_result/model_*/` (new format)
- `Train_result/seed_*/model/` (old format)
- Your specified path

### Q5: How to adjust model hyperparameters?
Recommended hyperparameter tuning order:
1. `hidden_dim`: 128 → 256 → 512
2. `num_layers`: 3 → 6 → 9
3. `learning_rate`: 0.001 → 0.0005 → 0.002
4. `dropout`: Try different dropout combinations

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please read CONTRIBUTING.md for contribution guidelines.

## 📧 Contact

For questions or suggestions, please contact us through:
- Submit an Issue
- Email project maintainers

## 🙏 Acknowledgments

Thanks to the following open-source projects for their support:
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [RDKit](https://www.rdkit.org/)
- [scikit-learn](https://scikit-learn.org/)

## 📚 References

If you use this project in your research, please cite the relevant papers:

```bibtex
@article{dyrk1a_gnn_2024,
    title={DYRK1A Activity Prediction Using Graph Neural Networks},
    author={Your Name},
    journal={Journal Name},
    year={2024}
}
``` 
