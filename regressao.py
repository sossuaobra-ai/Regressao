import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Previsão de Consumo Industrial",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("Este aplicativo utiliza um modelo de Regressão Linear treinado com dados sintéticos para prever o consumo de energia em tempo real com base em variáveis operacionais.")

# 1. Geração do Dataset Sintético (> 20 registros)
@st.cache_data
def gerar_dataset():
    np.random.seed(42)
    n_samples = 150  # Mais de 20 registros para robustez
    
    velocidade_esteira = np.random.uniform(1.0, 10.0, n_samples) # m/s
    maquinas_ativas = np.random.randint(1, 30, n_samples)         # unidades
    temp_ambiente = np.random.uniform(15.0, 40.0, n_samples)     # °C
    
    # Criando uma relação lógica com algum ruído gaussiano
    consumo_kwh = (
        40.0 
        + 12.5 * velocidade_esteira 
        + 18.2 * maquinas_ativas 
        + 2.5 * temp_ambiente 
        + np.random.normal(0, 4.0, n_samples)
    )
    
    df = pd.DataFrame({
        'Velocidade_Esteira': velocidade_esteira,
        'Maquinas_Ativas': maquinas_ativas,
        'Temp_Ambiente': temp_ambiente,
        'Consumo_kWh': consumo_kwh
    })
    return df

df_industrial = gerar_dataset()

# 2. Treinamento do Modelo de Regressão
@st.cache_resource
def treinar_modelo(df):
    X = df[['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']]
    y = df['Consumo_kWh']
    
    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns

modelo, features = treinar_modelo(df_industrial)

# 3. Controles na Barra Lateral (Sidebar)
st.sidebar.header("🎛️ Parâmetros Operacionais")
st.sidebar.markdown("Insira os valores atuais da fábrica:")

vel_input = st.sidebar.number_input(
    "Velocidade da Esteira (m/s)",
    min_value=float(df_industrial['Velocidade_Esteira'].min()),
    max_value=float(df_industrial['Velocidade_Esteira'].max()),
    value=float(df_industrial['Velocidade_Esteira'].mean()),
    step=0.1
)

maq_input = st.sidebar.number_input(
    "Máquinas Ativas",
    min_value=int(df_industrial['Maquinas_Ativas'].min()),
    max_value=int(df_industrial['Maquinas_Ativas'].max()),
    value=int(df_industrial['Maquinas_Ativas'].mean()),
    step=1
)

temp_input = st.sidebar.number_input(
    "Temperatura Ambiente (°C)",
    min_value=float(df_industrial['Temp_Ambiente'].min()),
    max_value=float(df_industrial['Temp_Ambiente'].max()),
    value=float(df_industrial['Temp_Ambiente'].mean()),
    step=0.5
)

# 4. Realizando a Previsão com os dados inseridos
dados_usuario = pd.DataFrame([[vel_input, maq_input, temp_input]], columns=features)
consumo_previsto = modelo.predict(dados_usuario)[0]

# 5. Layout da Tela Principal
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📊 Resultado da Previsão")
    st.markdown("Com base nos parâmetros configurados na barra lateral, a previsão de consumo energético é:")
    
    # Exibição em destaque usando st.metric
    st.metric(
        label="Consumo Energético Estimado", 
        value=f"{consumo_previsto:.2f} kWh"
    )
    
    with st.expander("🔍 Visualizar amostra do Dataset gerado"):
        st.dataframe(df_industrial.head(10), use_container_width=True)

with col2:
    st.subheader("📈 Contribuição das Variáveis (Coeficientes)")
    st.markdown("O gráfico abaixo ilustra o peso relativo de cada parâmetro no impacto do consumo final de energia.")
    
    # Gráfico de barras com Matplotlib representando os coeficientes
    fig, ax = plt.subplots(figsize=(6, 4))
    coeficientes = modelo.coef_
    
    barras = ax.barh(features, coeficientes, color=['#3182ce', '#dd6b20', '#38a169'])
    ax.set_xlabel("Coeficiente de Impacto (Peso no Modelo)")
    ax.set_title("Importância dos Fatores de Consumo")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Adicionando os valores nas barras
    for barra in barras:
        largura = barra.get_width()
        ax.text(
            largura + (0.5 if largura >= 0 else -1.5), 
            barra.get_y() + barra.get_height()/2, 
            f'{largura:.2f}', 
            va='center', 
            ha='left' if largura >= 0 else 'right', 
            fontsize=9,
            weight='bold'
        )
        
    st.pyplot(fig)
