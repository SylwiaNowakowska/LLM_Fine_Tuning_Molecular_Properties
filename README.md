# LLM_Fine_Tuning_Molecular_Properties

## Fine-tuning of ChemBERTa-2 for the HIV replication inhibition prediction.

 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chemberta-2-fine-tuning-for-molecules-hiv/molecular-property-prediction-on-hiv-dataset)](https://paperswithcode.com/sota/molecular-property-prediction-on-hiv-dataset?p=chemberta-2-fine-tuning-for-molecules-hiv)

This study is inspired by the DeepChem Tutorial "Transfer Learning with ChemBERTAa Transformers" [1], in which  ChemBERTa model, pre-trained on 77M SMILES from PubChem [2] was used for fine-tuning on molecules' toxicity task.
<br>
<br>
In this project, another version of the model, _i.e._ ChemBERTa-2 [3-5] is fine-tuned for HIV replication inhibition prediction (Fig. 1) using MoleculeNet Dataset [6]. Specifically, the influence of the pre-training method on the performance of the downstream task after fine-tuning is investigated. The model pre-trained with masked-language modeling (MLM) achieved better performance (AUROC 0.793), than the model pre-trained with multi-task regression (MTR) (AUROC 0.733). The alterations in the distributions of molecular embeddings before and after fine-tuning highlight the improved capacity of models to distinguish between active and inactive HIV molecules. 
<br>
[S. Nowakowska, ChemBERTa-2: Fine-Tuning for Molecule’s HIV Replication Inhibition Prediction](https://chemrxiv.org/engage/chemrxiv/article-details/65030b55b338ec988a780108) <br>
<br>
<br>
![Alt text](https://github.com/SylwiaNowakowska/LLM_Fine_Tuning_Molecular_Properties/blob/main/Fig_1_Study_Design.png)
<br> 
Fig. 1) Study design
<br>
<br>
<br>
![Alt text](https://github.com/SylwiaNowakowska/LLM_Fine_Tuning_Molecular_Properties/blob/main/Fig_2_MLM_MTR_Model_Performance.png)
<br> 
Fig. 2) Models' performance
<br>
<br>
<br>
![Alt text](https://github.com/SylwiaNowakowska/LLM_Fine_Tuning_Molecular_Properties/blob/main/Fig_3_MLM_MTR_Embeddings.png)
<br> 
Fig. 3) Latent representations of the embeddings of the molecules contained in the test set for both MLM and MTR models prior and after fine-tuning
<br>
<br> References: <br>
[1] [Transfer Learning with ChemBERTAa turorial](https://colab.research.google.com/github/deepchem/deepchem/blob/master/examples/tutorials/Transfer_Learning_With_ChemBERTa_Transformers.ipynb) <br>
[2] [S. Chithrananda _et al._, ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/pdf/2010.09885.pdf) <br>
[3] [W. Ahmad _et al._, ChemBERTa-2: Towards Chemical Foundation Models](https://arxiv.org/pdf/2209.01712.pdf) <br>
[4] [HuggingFace, DeepChem: ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) <br>
[5] [HuggingFace, DeepChem: ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR) <br>
[6] [MoleculeNet Dataset](https://moleculenet.org/datasets-1) <br>


