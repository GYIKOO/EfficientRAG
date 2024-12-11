# EfficientRAG-Pro CS291A Project
This is a final project of UCSB CS291A 2024 Fall.

To run this project, you need to set up following the original EfficientRAG.

## Setup

### Installation

You need to install PyTorch >= 2.1.0 first, and then install dependent Python libraries by running the command

```bash
pip install -r requirements.txt
```

You can also create a conda environment with python>=3.9

```bash
conda create -n <ENV_NAME> python=3.9 pip
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Preparation

1. Both datasets (2WikiMQA and MuSiQue) used in our experiments have been preprocessed and stored under `data`. 

2. Download the retriever model [Contriever](https://huggingface.co/facebook/contriever-msmarco) and base model [DeBERTa](https://huggingface.co/microsoft/deberta-v3-large), put them under `model_cache`

3. Deploy [LLaMA-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) with [vLLM](https://github.com/vllm-project/vllm) framework, and configure it in `src/language_models/llama.py`

## Training

We will use the MuSiQue dataset as an example. You could train on 2WikiMQA in the same way.
        
Training Filter model

```bash
python src/efficient_rag/filter_training.py \
    --dataset musique \
    --save_path saved_models/filter
```

Training Labeler model

```bash
python src/efficient_rag/labeler_training.py \
    --dataset musique \
    --tags 2
```

## Inference

EfficientRAG retrieve procedure

```bash
python src/efficientrag_retrieve.py \
    --dataset musique \
    --retriever contriever \
    --labels 2 \
    --labeler_ckpt <<PATH_TO_LABELER_CKPT>> \
    --filter_ckpt <<PATH_TO_FILTER_CKPT>> \
    --topk 10 \
```

TextRank Pruning

```bash
python src/textrank.py --fpath <<MODEL_INFERENCE_RESULT>> --top_k 10
```
You can try different top_k to control the pruning proportion.


Use LLaMA-3-8B-Instruct as generator
```bash
python src/efficientrag_qa.py \
    --fpath <<MODEL_INFERENCE_RESULT_AFTER_PRUNING>> \
    --model llama-8B \
    --dataset musique
```

## Evaluation
Retrieve results
```bash
python src/evaluation/retrieve.py --fpath <<QA_RESULT>>
```

Correctness
```bash
python src/evaluation/correctness.py \
    --fpath <<QA_RESULT>>
    --model llama-8b-instruct
```

## Citation

If you find this paper or code useful, please cite by:

```txt
@inproceedings{zhuang2024efficientrag,
  title={EfficientRAG: Efficient Retriever for Multi-Hop Question Answering},
  author={Zhuang, Ziyuan and Zhang, Zhiyang and Cheng, Sitao and Yang, Fangkai and Liu, Jia and Huang, Shujian and Lin, Qingwei and Rajmohan, Saravan and Zhang, Dongmei and Zhang, Qi},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={3392--3411},
  year={2024}
}
```
