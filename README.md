# CausalMed

The official version of `CausalMed` includes the complete model architecture, training methods, and test examples.
https://dl.acm.org/doi/abs/10.1145/3627673.3679542

## 1. Folder Specification


- `data/`
  - `input/` 
    - `drug-atc.csv`, `ndc2atc_level4.csv`, `ndc2rxnorm_mapping.txt`: mapping files for drug code transformation
    - `idx2ndc.pkl`: It maps ATC-4 code to rxnorm code and then query to drugbank.
    - `idx2drug.pkl`: Drug ID (we use ATC-4 level code to represent drug ID) to drug SMILES string dictionary
      
  - `output/`
    - `voc_final.pkl`: diag/prod/med index to code dictionary
    - `ddi_A_final.pkl`: ddi adjacency matrix
    - `ddi_matrix_H.pkl`: H mask structure (This file is created by ddi_mask_H.py)
    - `records_final.pkl`: The final diagnosis-procedure-medication EHR records of each patient. Due to policy reasons, we are unable to provide processed data. Users are asked to process it themselves according to the instructions in the next section
      
  - `graphs/`
    - `causal_graph.pkl`: casual graphs in DAG form
    - `Diag_Med_causal_effect.pkl`,`Proc_Med_casual_effect.pkl`: causal effects between diag/proc and med
    
  - `ddi_mask_H.py`: The python script responsible for generating `ddi_mask_H.pkl` and `substructure_smiles.pkl`.
  - `processing.py`: The python script responsible for generating `voc_final.pkl`, `records_final.pkl`, and `ddi_A_final.pkl`   

- `src/` folder contains all the source code.
  - `modules/`: Code for model definition.
  - `util.py`: Code for metric calculations and some data preparation.
  - `training.py`: Code for the functions used in training and evaluation.
  - `main.py`: Train or evaluate our Model.
 
- `saved/` 
  - `trained_model`:  test example, a model we have trained. Users can directly check using the test mode
  - `parameter_report.txt`: Log file containing all parameters
  
**Note1:** `data/` only contains part of the data. See the [Data processing] section for more details.

**Note2:** Due to some relatively complex environment dependencies during the causal graph generation phase, for the convenience of users in studying or validating our work, we have submitted a file named `causal_construction_easyuse.py`. This file can be used in conjunction with the already generated causal graphs, replacing the `causal_construction.py` file. While this method is more convenient, we strongly recommend researchers to retrain the causal graphs to ensure rigor.

## 2. Operation

## 2.1 Package Dependency

Please install the environment according to the following version

```bash
python == 3.8.17
torch == 2.0.1
dill == 0.3.6
numpy == 1.22.3
pandas == 2.0.2 
torch-geometric == 2.3.1
cdt == 0.6.0
dowhy ==  0.10.1
statsmodels == 0.14.0
```
## 2.2 Data Processing

1.MIMIC-III:Due to the privacy of medical data, we cannot directly provide source data. You must apply for permission at https://physionet.org/content/mimiciii/1.4/ and download the data set after passing the review. And go into the folder and unzip three main files (PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz, DIAGNOSES_ICD.csv.gz) into /data/inputs/.

2.MIMIC-IV:Similar to the MIMIC-IV process

3.Known DDI:download the DDI file and move it to the data folder download https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing

4.processing the data to get a complete records

```bash
python data/processing.py
python data/ddi_mask_H.py
```

### 2.3 Run the Code

```bash
python src/main.py
```


## 3. Citation & Acknowledgement

We are grateful to everyone who contributed to this project. The article has been published in the CIKM2024.

If the code and the paper are useful for you, it is appreciable to cite our paper:
```bash
@inproceedings{li2024causalmed,
  title={CausalMed: Causality-Based Personalized Medication Recommendation Centered on Patient Health State},
  author={Li, Xiang and Liang, Shunpan and Lei, Yu and Li, Chen and Hou, Yulei and Zheng, Dashun and Ma, Tengfei},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={1276--1285},
  year={2024}
}

```
