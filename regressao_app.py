import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuração da Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Painel de Previsão de Consumo de Energia",
    layout="wide"
)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.write("Insira os parâmetros na barra lateral para prever o consumo energético industrial (kWh).")

# ---------------------------------------------------------
# 1. Geração do Dataset
# ---------------------------------------------------------
@st.cache_data
def gerar_dados():
    np.random.seed(42)
    n_samples = 50
    
    velocidade = np.random.uniform(1.0, 10.0, n_samples)          # m/s
    maquinas = np.random.randint(1, 15, n_samples)                 # Quantidade
    temp_ambiente = np.random.uniform(18.0, 38.0, n_samples)       # °C
    
    # Relação matemática realista com um pouco de ruído
    consumo = (velocidade * 15.5) + (maquinas * 42.0) + (temp_ambiente * 8.2) + np.random.normal(0, 15, n_samples)
    
    df = pd.DataFrame({
        'Velocidade_Esteira': velocidade,
        'Maquinas_Ativas': maquinas,
        'Temp_Ambiente': temp_ambiente,
        'Consumo_kWh': consumo
    })
    return df

df = gerar_dados()

# ---------------------------------------------------------
# 2. Treinamento do Modelo de Regressão
# ---------------------------------------------------------
X = df[['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']]
y = df['Consumo_kWh']

modelo = LinearRegression()
modelo.fit(X, y)

# ---------------------------------------------------------
# 3. Barra Lateral (Sidebar) - Entradas do Usuário
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parâmetros de Entrada")

vel_input = st.sidebar.slider(
    "Velocidade da Esteira (m/s)",
    min_value=float(df['Velocidade_Esteira'].min()),
    max_value=float(df['Velocidade_Esteira'].max()),
    value=5.0,
    step=0.1
)

maq_input = st.sidebar.number_input(
    "MÁQUINAS Ativas",
    min_value=int(df['Maquinas_Ativas'].min()),
    max_value=int(df['Maquinas_Ativas'].max()),
    value=8,
    step=1
)

temp_input = st.sidebar.slider(
    "Temperatura Ambiente (°C)",
    min_value=float(df['Temp_Ambiente'].min()),
    max_value=float(df['Temp_Ambiente'].max()),
    value=25.0,
    step=0.5
)

# ---------------------------------------------------------
# 4. Previsão do Consumo
# ---------------------------------------------------------
entrada_usuario = pd.DataFrame({
    'Velocidade_Esteira': [vel_input],
    'Maquinas_Ativas': [maq_input],
    'Temp_Ambiente': [temp_input]
})

previsao = modelo.predict(entrada_usuario)[0]

# ---------------------------------------------------------
# 5. Exibição dos Resultados no Painel Principal
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Estimativa de Consumo")
    st.metric(
        label="Consumo Previsto",
        value=f"{previsao:.2f} kWh"
    )
    
    st.divider()
    
    with st.expander("Ver primeiras linhas do Dataset"):
        st.dataframe(df.head(10))

with col2:
    st.subheader("📈 Contribuição dos Coeficientes do Modelo")
    
    # Coeficientes do modelo representando o peso de cada variável
    coeficientes = modelo.coef_
    variaveis = ['Velocidade Esteira', 'Máquinas Ativas', 'Temp. Ambiente']
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    barras = ax.barh(variaveis, coeficientes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xlabel('Coeficiente (Aumento de kWh por Unidade)')
    ax.set_title('Impacto de cada Variável no Consumo')
    
    # Adicionar os valores exatos nas barras
    for barra in barras:
        largura = barra.get_width()
        ax.annotate(f'{largura:.2f}',
                    xy=(largura, barra.get_y() + barra.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

    # Ajuste do limite horizontal para legibilidade das anotações
    ax.set_xlim(0, max(coeficientes) * 1.25)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
