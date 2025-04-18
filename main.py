#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para análise de dados do INCC (Índice Nacional de Custo da Construção)
"""

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
import plotly.express as px
import plotly.graph_objects as go

def main():
    # Lendo o arquivo de dados
    df = pd.read_csv('./data/dataset INCC.csv', sep="\t")
    
    # Exibindo as primeiras linhas do DataFrame
    print("Primeiras linhas do DataFrame:")
    print(df.head())
    
    # Informações básicas sobre o DataFrame
    print("\nInformações do DataFrame:")
    print(df.info())
    
    # Convertendo a coluna 'INCC Geral' para float
    df['INCC Geral float'] = df['INCC Geral'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
    
    # Removendo colunas desnecessárias
    df.drop(columns=['INCC Geral'], inplace=True)
    
    print("\nDataFrame após processamento:")
    print(df.head())

if __name__ == "__main__":
    main() 