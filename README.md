# (UNIRIO 2025) Trabalho final para a disciplina de Deep Learning, utilizando o dataset FairFace

Para criar e ativar o ambiente
```bash
pyenv local 3.10.12
pyenv virtualenv 3.10.12 fairface-env
pyenv local fairface-env
```

Para instalar as dependências do projeto
```bash
pip install -r requirements.txt
```

Crie o kernel do jupyter a partir do ambiente atual
```bash
python -m ipykernel install --user --name fairface-env --display-name "Python (fairface)"
```

Para utilizar este repositório, é preciso baixar o dataset através do link abaixo:
https://www.kaggle.com/datasets/aibloy/fairface
Após download do zip, extraí-lo dentro da pasta /data
