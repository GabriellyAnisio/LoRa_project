# Finetuning de um Modelo de Linguagem com LoRA

# Este script realiza o finetuning de um modelo de linguagem utilizando LoRA.

# Importação das bibliotecas necessárias
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

# Função de instalação das bibliotecas
def install_dependencies():
    os.system('pip install -q bitsandbytes datasets accelerate loralib')
    os.system('pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git')
    os.system('pip install transformers torch torchvision')

# Instalação das bibliotecas necessárias
install_dependencies()

# Autenticação no Hugging Face Hub
notebook_login()

# Verificação das GPUs disponíveis
os.system('nvidia-smi -L')

# Configuração do ambiente para utilizar a GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Carregamento do modelo pré-treinado e do tokenizador
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m",
    device_map='auto',  # Mapeia automaticamente para múltiplas GPUs se disponível
    force_download=True
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

# Congelamento de parâmetros para evitar o treinamento de todo o modelo
for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # Reduz o número de ativações armazenadas
model.enable_input_require_grads()

# Classe para converter a saída para o formato float32
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

model.lm_head = CastOutputToFloat(model.lm_head)

# Função para contar e imprimir os parâmetros treináveis
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# Configuração e aplicação de LoRA
config = LoraConfig(
    r=16,  # Cabeças de atenção
    lora_alpha=32,  # Escala de alpha
    target_modules=["query_key_value"],  # Módulos alvo
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # Tipo de tarefa
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Carregamento do dataset e preparação dos dados
data = load_dataset("Abirate/english_quotes")

def merge_columns(example):
    example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
    return example

train_dataset = data['train']
train_dataset = train_dataset.map(merge_columns)
train_dataset = train_dataset.map(lambda samples: tokenizer(samples['prediction']), batched=True)

# Configuração e treinamento do modelo
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,
    warmup_steps=1,
    max_steps=1,
    learning_rate=2e-4,
    logging_steps=1,
    output_dir='outputs'
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # Silencia os avisos. Reative para inferência!

try:
    trainer.train()
except KeyboardInterrupt:
    print("Interrupção manual!")

# Envio do modelo para o Hugging Face Hub
model.push_to_hub(
    "tayyibsupercool/bloom-560m-lora",
    use_auth_token=True,
    commit_message="basic training",
    private=True
)

# Carregamento do modelo finetunado
peft_model_id = "tayyibsupercool/bloom-560m-lora"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

batch = tokenizer("I like strawberries ->: ", return_tensors='pt')

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
