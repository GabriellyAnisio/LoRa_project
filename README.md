# Projeto de Fine-Tuning de Modelos Transformers

Este projeto tem como objetivo realizar o fine-tuning de modelos pré-treinados da biblioteca Hugging Face Transformers para uma tarefa específica de modelagem de linguagem, utilizando um dataset de citações em inglês. O modelo utilizado neste exemplo é o `bigscience/bloom-560m`, mas você pode substituir por qualquer outro modelo disponível na biblioteca.

REFERÊNCIA: [LoRA: Low Rank Adaptation of Large Language Models](https://medium.com/@tayyibgondal2003/loralow-rank-adaptation-of-large-language-models-33f9d9d48984)

## Requisitos

Antes de começar, certifique-se de que você tenha instalado as seguintes dependências:

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Datasets

Você pode instalar todas as dependências necessárias utilizando o `pip`:

```bash
pip install torch transformers datasets
```

## Estrutura do Projeto

```
|-- src
|   |-- LoRA.ipynb        # Notebook principal para realizar o fine-tuning, detalhado e explicado
|   |-- LoRA.py           # Script Principal
|   |-- README.md             # Descrição dos dados utilizados (opcional)
|-- outputs                   # Diretório onde os resultados e modelos treinados serão salvos
|-- README.md                 # Este arquivo README
```

## Dataset

O dataset utilizado neste projeto é o `Abirate/english_quotes`, disponível no Hugging Face Datasets. O dataset consiste em uma coleção de citações em inglês com tags associadas.

## Fine-Tuning do Modelo

O script principal, `fine_tuning.py`, realiza as seguintes etapas:

1. **Carregamento do Modelo e Tokenizer:** Utiliza o modelo `bigscience/bloom-560m` e o tokenizer correspondente.
2. **Preparação do Dataset:** O dataset é carregado e processado, combinando as citações e tags em uma única string.
3. **Configuração do Treinamento:** As configurações de treinamento são definidas, incluindo a taxa de aprendizado, o tamanho do batch, e outros parâmetros relevantes.
4. **Treinamento do Modelo:** O modelo é treinado usando o dataset processado e as configurações definidas.

### Executando o Fine-Tuning

Para realizar o fine-tuning do modelo, execute o script `fine_tuning.py`:

```bash
python src/fine_tuning.py
```

Os resultados do treinamento, incluindo o modelo ajustado, serão salvos no diretório `outputs`.

## Possíveis Erros e Soluções

Durante a execução do projeto, você pode encontrar alguns erros comuns:

- **NotImplementedError: Cannot copy out of meta tensor; no data!**: Esse erro pode ocorrer se você estiver utilizando um modelo quantizado ou se o ambiente não estiver corretamente configurado. Verifique se você está utilizando um modelo não quantizado e se as bibliotecas estão atualizadas.

- **Problemas com a Configuração da GPU:** Certifique-se de que a GPU está configurada corretamente, se estiver usando uma. Caso contrário, experimente rodar o treinamento na CPU.

## Personalização

Você pode personalizar o projeto de diversas formas:

- **Substituir o Modelo:** Altere o modelo pré-treinado utilizado no script para outro disponível no Hugging Face Models.
- **Modificar o Dataset:** Utilize outro dataset de sua escolha, ajustando o código para processá-lo corretamente.
- **Ajustar Hiperparâmetros:** Modifique os parâmetros de treinamento, como a taxa de aprendizado e o tamanho do batch, para otimizar o desempenho do modelo.