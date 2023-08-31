# LLM_Fine_Tuning_Molecular_Properties

## Fine-tuning of ChemBERTa-2 for the HIV replication inhibition prediction.



This notebook is based on DeepChem Tutorial "Transfer Learning with ChemBERTAa Transformers" [1]. <br>
In the tutorial ChemBERTa model, pre-trained on 77M SMILES from PubChem [2] is used for fine-tuning on molecules' toxicity task.
<br>
<br>
In this notebook, another version, i.e. ChemBERTa-2 model [3,4] is used for fine-tuning the molecules' ability to inhibit HIV task [5]. <br><br>
The leaderboard for this dataset can be found here: [Molecular Property Prediction on HIV dataset](https://paperswithcode.com/sota/molecular-property-prediction-on-hiv-dataset)

<br><br>
<br> References: <br>
[1] [Transfer Learning with ChemBERTAa turorial](https://colab.research.google.com/github/deepchem/deepchem/blob/master/examples/tutorials/Transfer_Learning_With_ChemBERTa_Transformers.ipynb) <br>
[2] [S. Chithrananda et al., ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/pdf/2010.09885.pdf) <br>
[3] [W. Ahmad et al., ChemBERTa-2: Towards Chemical Foundation Models](https://arxiv.org/pdf/2209.01712.pdf) <br>
[4] [HuggingFace, DeepChem: ChemBERTa-77M-MLM ](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) <br>
[5] [MoleculeNet Dataset](https://moleculenet.org/datasets-1) <br>
