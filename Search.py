import numpy as np
import os
import pandas as pd
import urllib.request
import faiss
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TFAutoModel
import torch
import re
from datasets import Dataset
import pickle
from collections import defaultdict
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 필수 모듈 임포트
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
# from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
# import tensorflow as tf

# SBERT 모델 설정
model_ckpt = 'BM-K/KoSimCSE-roberta-multitask'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)

# # SBERT 모델 설정
# model_ckpt = 'BM-K/KoSimCSE-roberta-multitask'
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = AutoModel.from_pretrained(model_ckpt)

# 데이터셋 로드 및 인덱스 설정
loaded_dataset = load_from_disk("./Data/faiss_indexed_dataset")
loaded_dataset.add_faiss_index(column="embeddings")

# 임베딩 생성 함수
def get_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="tf")
    embeddings = model(**inputs).last_hidden_state[:, 0, :]  # CLS token 사용
    return embeddings

# 유사 문서 검색 함수
def search_similar_documents(query, k=30):
    # 쿼리 임베딩 생성
    query_embedding = get_embeddings([query]).numpy()
    
    # 유사 문서 검색
    scores, samples = loaded_dataset.get_nearest_examples(
        "embeddings", query_embedding, k=k
    )
    
    # 결과를 DataFrame으로 변환하고 정렬
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["유사도"] = scores
    samples_df.sort_values("유사도", ascending=False, inplace=True)
    
    return samples_df

# 키워드 포함 여부 확인 함수
def filter_by_keyword(samples_df, keyword):
    # 쿼리문을 공백으로 분할하여 키워드 리스트 생성
    keywords = keyword.split()
    
    # 키워드 포함 여부 확인
    def keyword_match(text, keywords):
        return any(keyword in text for keyword in keywords)
    
    # DataFrame에 키워드 포함 여부 열 추가
    samples_df["키워드_포함"] = samples_df["processed"].apply(lambda x: keyword_match(x, keywords))
    
    # 유사도 및 키워드 포함 여부 기준 정렬
    samples_df.sort_values(by=["키워드_포함", "유사도"], ascending=[False, False], inplace=True)
    samples_df.reset_index(drop=True, inplace=True)
    
    return samples_df

# 전체 검색 및 필터링 함수
def search_documents_with_filter(query, k=30):
    # 유사 문서 검색
    results_df = search_similar_documents(query, k)
    
    # 키워드 필터링 및 정렬
    filtered_df = filter_by_keyword(results_df, query)
    
    return filtered_df