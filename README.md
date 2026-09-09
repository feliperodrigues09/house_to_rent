# 🏠 House Rent Pricing | EDA, Market Segmentation & Stacking

> **Como a Análise de Dados e Machine Learning podem identificar gargalos e apoiar decisões em um processo operacional?** Este projeto desenvolve um MVP analítico para apoiar a precificação de imóveis por meio de qualidade de dados, segmentação de mercado e uma arquitetura de Stacking com modelos especialistas.

---

## Contexto

A precificação inicial é uma das etapas mais críticas da captação de imóveis. Quando o valor sugerido está desalinhado com o mercado, a operação enfrenta negociações mais difíceis, decisões inconsistentes e menor confiança na avaliação do corretor.

O desafio deste projeto foi investigar **como dados e Machine Learning podem apoiar essa decisão**, utilizando uma base de mais de **6 mil imóveis brasileiros** em um mercado altamente heterogêneo, onde imóveis estruturalmente semelhantes podem apresentar diferenças significativas de preço.

Em vez de buscar apenas o algoritmo mais sofisticado, a solução foi construída sobre três pilares:

- **Qualidade dos dados** para reduzir ruído e inconsistências;
- **Segmentação de mercado** para representar a heterogeneidade da base;
- **Especialização de modelos**, combinando um modelo global com especialistas por segmento através de Stacking.

---

# 🎯 Objetivos

O projeto foi estruturado para responder quatro perguntas de negócio:

1. Quais segmentos de imóveis compõem a base?
2. Quais características mais influenciam o valor do aluguel?
3. É possível construir um modelo de apoio à precificação durante a captação?
4. Onde e por que o modelo erra?

Mais do que prever preços, o objetivo foi compreender **os limites da informação disponível** e como eles impactam a tomada de decisão.

# Arquitetura da solução

A solução foi desenvolvida como uma pipeline completa de Data Science.

```text
Business Understanding
      │
      ▼
Dados Brutos
      │
      ▼
Limpeza + Qualidade dos Dados
      │
      ▼
Engenharia de Features
      │
      ▼  
Segmentação de Mercado (K-Means)
      │
      ▼
     EDA
      │
      ▼
Classificação do Segmento
      │
      ▼
┌──────────────────────────────────────────┐
│            Arquitetura Stacking          │
│                                          │
│   Modelo Global                          │
│          │                               │
│          ├───────────────┐               │
│          ▼               ▼               │
│   Especialista GPAP  Especialista GPBP   │
│          │               │               │
│   Especialista PPAP  Especialista PPBP   │
│          └───────┬───────┘               │
│                  ▼                       │
│          Ridge Meta-Model                │
└──────────────────────────────────────────┘
                  │
                  ▼
          Valor estimado do aluguel
                  │
                  ▼
              Conclusão
```

Essa arquitetura permite que cada especialista aprenda padrões específicos do seu segmento, enquanto o meta-modelo aprende **quando confiar no modelo global e quando priorizar um especialista**.

---

# Qualidade dos dados

### Dicionário de Dados: `houses_to_rent`

| Coluna                            | Tradução / Significado                                             | Tipo de Dado Padrão |
| :-------------------------------- | :------------------------------------------------------------------- | :------------------- |
| **`city`**                | Cidade onde o imóvel está localizado                               | Categórica (String) |
| **`area`**                | Área total do imóvel em metros quadrados (m²)                     | Numérica (Int)      |
| **`rooms`**               | Quantidade de quartos                                                | Numérica (Int)      |
| **`bathroom`**            | Quantidade de banheiros                                              | Numérica (Int)      |
| **`parking spaces`**      | Quantidade de vagas de garagem                                       | Numérica (Int)      |
| **`floor`**               | Andar do imóvel                                                     | Categórica (String) |
| **`animal`**              | Indica se o imóvel aceita animais (`acept` / `not acept`)       | Categórica (String) |
| **`furniture`**           | Indica se o imóvel é mobiliado (`furnished` / `not furnished`) | Categórica (String) |
| **`hoa (R$)`**            | Valor da taxa de condomínio (Homeowners Association Tax)            | Numérica / Texto    |
| **`rent amount (R$)`**    | Valor mensal do aluguel                                              | Numérica / Texto    |
| **`property tax (R$)`**   | Valor do IPTU                                                        | Numérica / Texto    |
| **`fire insurance (R$)`** | Valor do seguro contra incêndio                                     | Numérica (Float)    |
| **`total (R$)`**          | Custo total mensal (soma do aluguel e taxas)                         | Numérica / Texto    |

A etapa mais importante do projeto não foi a modelagem, mas a investigação da qualidade da base.

Foram identificados problemas como:

- inconsistências entre aluguel, condomínio, IPTU e valor total;
- registros com custos embutidos de formas diferentes;
- outliers financeiros;
- variáveis com alto potencial de introduzir viés.

Os outliers não foram removidos automaticamente. Cada grupo extremo foi investigado para distinguir **erros de coleta**, **casos legítimos** e **imóveis raros**, preservando a representatividade da base.

### Principais entregas

- Tratamento de inconsistências financeiras ocultas
- Engenharia de variáveis derivadas (`valor_m²`, proporção de condomínio, indicadores binários)
- Padronização para modelagem sem vazamento de informação

---

# Segmentação de mercado

O mercado imobiliário não é homogêneo.

Aplicar um único modelo sobre toda a base significa assumir que apartamentos compactos e imóveis de alto padrão seguem a mesma lógica de formação de preço, uma hipótese que a EDA mostrou não ser verdadeira.

A clusterização permitiu identificar quatro segmentos operacionais:

| Segmento                              | Interpretação                           |
| ------------------------------------- | ----------------------------------------- |
| Grande Porte • Alto Posicionamento   | Imóveis grandes com maior valor relativo |
| Grande Porte • Baixo Posicionamento  | Imóveis grandes mais acessíveis         |
| Pequeno Porte • Alto Posicionamento  | Imóveis compactos premium                |
| Pequeno Porte • Baixo Posicionamento | Imóveis compactos econômicos            |

### Principal insight

Imóveis com características construtivas muito semelhantes apresentavam **valores por metro quadrado mais que duas vezes diferentes**, indicando que variáveis ausentes, especialmente localização e conservação, exercem papel decisivo na precificação.

---

# Classificação dos segmentos

Antes da regressão, foi necessário criar um classificador capaz de atribuir novos imóveis ao segmento correto.

Foram comparados Logistic Regression, Decision Tree e Random Forest.

A **Logistic Regression** foi escolhida por apresentar o melhor equilíbrio entre desempenho e interpretabilidade, alcançando aproximadamente **74% de acurácia** na classificação dos segmentos.

Essa etapa permitiu transformar a segmentação em um componente utilizável para novos imóveis, tornando a arquitetura escalável para inferência.

---

# Modelo de precificação

O núcleo do projeto é uma arquitetura **Global-Local Ensemble**.

Em vez de utilizar um único modelo para todo o mercado:

- um **modelo global** aprende tendências gerais;
- **quatro especialistas** aprendem padrões específicos de cada segmento;
- um **meta-modelo Ridge** combina dinamicamente essas previsões.

### Controle de Data Leakage

Um dos cuidados técnicos do projeto foi impedir que o meta-modelo aprendesse a partir de previsões produzidas por modelos que já haviam visto aquelas mesmas observações.

Para isso, todas as predições utilizadas no Stacking foram geradas por **Out-of-Fold Predictions (OOF)** com validação cruzada, garantindo uma avaliação mais fiel da capacidade de generalização.

---

# Resultados

## Regressão (Stacking)

| Métrica       |              Teste |
| -------------- | -----------------: |
| **R²**  |    **0,811** |
| **MAE**  | **R$ 1.332** |
| **RMSE** | **R$ 2.096** |

## Principais descobertas

- A especialização reduziu o erro em segmentos onde o modelo global apresentava maior dificuldade.
- Decision Trees, quando organizadas em uma arquitetura inteligente, entregaram desempenho competitivo sem exigir modelos extremamente complexos.
- O principal limitador da performance não foi o algoritmo, mas **a ausência de variáveis críticas** na coleta de dados.

---

# 💡 Arquitetura Inteligente > Força Bruta

Um dos aprendizados centrais deste projeto foi perceber que **mais complexidade não significa necessariamente uma solução melhor**.

Embora modelos como Random Forest e XGBoost sejam frequentemente priorizados em problemas de regressão, este estudo mostrou que uma arquitetura baseada em:

- engenharia de features,
- segmentação,
- modelos especialistas,
- e um meta-modelo simples,

é capaz de extrair excelente desempenho mantendo baixo custo computacional e alta interpretabilidade.

O ganho veio da **organização dos modelos**, não apenas da escolha do algoritmo.

---

# Visão de Produto

Este projeto deve ser interpretado como um **MVP analítico**, e não como um sistema pronto para produção.

Em um cenário real, a evolução natural seria implementar o modelo em **Shadow Mode**:

```text
Captação do imóvel
        │
        ▼
 Avaliação do corretor
        │
        ├───────────────┐
        │               │
        ▼               ▼
 Precificação humana  Modelo Stacking
        │               │
        └────── Comparação ──────┘
                │
                ▼
      Validação operacional
```

Esse piloto permitiria medir aderência, identificar divergências e justificar investimentos futuros em governança e enriquecimento da coleta antes de automatizar decisões.

---

# Tecnologias

**Linguagem e bibliotecas**

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

**Técnicas utilizadas**

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Outlier Analysis
- Correlation Analysis
- K-Means Clustering
- Logistic Regression
- Decision Trees
- Ridge Regression
- Cross Validation
- Out-of-Fold Predictions
- Stacking Ensemble

---

# 📂 Estrutura do projeto

```text
.
├── Dataset/
│   └── houses_to_rent.csv
│
├── Images/ 
│
├── Notebooks/
│   └── house_to_rent.ipynb
│
├── Reports/
│   └── house_to_rent.html
│
├── environment.yml     # Ambiente Conda para projeto
├── .gitignore          # Arquivos ignorados
└── README.md           # Este arquivo
```

## Como reproduzir este projeto

**Pré-requisitos:** Certifique-se de ter o [Git](https://git-scm.com/) e o gerenciador de pacotes [Conda](https://docs.conda.io/en/latest/) (Anaconda ou Miniconda) instalados em sua máquina.

**1. Clone o repositório**
Abra o seu terminal e execute:

```bash
git clone [https://github.com/feliperodrigues09/house_to_rent.git](https://github.com/feliperodrigues09/house_to_rent.git)
cd NOME_DO_REPOSITORIO
```

---

# Competências demonstradas

Este projeto reúne competências de **Data Analytics, Data Science e visão de produto**, incluindo:

### Analytics

- Formulação de perguntas de negócio
- Investigação de qualidade de dados
- EDA orientada à decisão
- Análise de erros e interpretação de padrões

### Machine Learning

- Clusterização
- Classificação
- Regressão
- Ensemble Learning
- Validação cruzada
- Prevenção de Data Leakage

### Produto & Operações

- Tradução de problemas operacionais em soluções analíticas
- Avaliação crítica das limitações da base
- Proposição de MVP e estratégia de validação em ambiente real

---

# Conclusão

Este projeto começou como um problema de precificação, mas revelou uma questão mais ampla: **dados só geram valor quando conseguem apoiar decisões reais**.

Ao combinar qualidade dos dados, segmentação e especialização de modelos, a solução demonstra que uma arquitetura bem desenhada pode ser mais eficaz do que simplesmente aumentar a complexidade do algoritmo.

O resultado é um MVP que evidencia não apenas capacidade de modelagem, mas também uma abordagem orientada à operação, à interpretabilidade e à evolução contínua do produto de dados.
