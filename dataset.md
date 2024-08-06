### Raccolta del Dataset del Codice Civile Italiano

Per creare un dataset del Codice Civile Italiano per il fine-tuning di un modello AI, è necessario raccogliere il testo integrale del Codice Civile e altre risorse legali rilevanti. Ecco i passaggi dettagliati per ottenere e preparare questi dati.

### Fonti di Dati

1. **Gazzetta Ufficiale**
   - La Gazzetta Ufficiale pubblica tutte le leggi e i decreti ufficiali dello Stato italiano.
   - Puoi accedere agli archivi della Gazzetta Ufficiale online per scaricare il testo completo del Codice Civile.
   - **URL**: [Gazzetta Ufficiale](https://www.gazzettaufficiale.it/)

2. **Normattiva**
   - Normattiva è un portale che offre accesso gratuito ai testi normativi aggiornati.
   - Puoi cercare il Codice Civile e altre normative rilevanti e scaricarne i testi.
   - **URL**: [Normattiva](http://www.normattiva.it/)

3. **EUR-Lex**
   - EUR-Lex fornisce accesso alla legislazione dell'Unione Europea e può contenere riferimenti utili alle normative italiane.
   - **URL**: [EUR-Lex](https://eur-lex.europa.eu/)

4. **Siti Accademici e Biblioteche Digitali**
   - Alcuni siti accademici e biblioteche digitali offrono accesso ai testi legali per scopi di ricerca.
   - Verifica se la tua università o istituzione ha accesso a queste risorse.

### Pre-elaborazione dei Dati

#### Passaggi di Pre-elaborazione

1. **Scaricamento dei Testi**
   - Scarica il testo del Codice Civile italiano da una delle fonti sopra elencate.
   - Assicurati che il formato dei file sia testuale (es. TXT, PDF convertito in TXT).

2. **Pulizia del Testo**
   - Rimuovi elementi non necessari come intestazioni, numeri di pagina, e note a piè di pagina.
   - Se necessario, converti il testo da PDF a formato testuale utilizzando strumenti come `pdftotext`.

3. **Formattazione**
   - Organizza il testo in modo che sia facilmente gestibile per il fine-tuning. Ad esempio, suddividi il testo in articoli e paragrafi.
   - Utilizza strumenti di elaborazione del testo come `spaCy` o `NLTK` per tokenizzare il testo.

### Esempio di Script per la Pulizia e la Formattazione

```python
import re

def clean_text(text):
    # Rimuove numeri di pagina e note a piè di pagina
    text = re.sub(r'\d+', '', text)
    # Rimuove spazi e linee bianche multiple
    text = re.sub(r'\s+', ' ', text)
    return text

# Leggi il file di testo del Codice Civile
with open('codice_civile.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

# Pulisci il testo
cleaned_text = clean_text(raw_text)

# Scrivi il testo pulito in un nuovo file
with open('codice_civile_cleaned.txt', 'w', encoding='utf-8') as file:
    file.write(cleaned_text)
```

### Fine-Tuning del Modello

Utilizza il testo pulito del Codice Civile per il fine-tuning del modello AI. Ecco come configurare il processo di fine-tuning utilizzando la libreria `transformers` di Hugging Face.

### Esempio di Codice per il Fine-Tuning

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Carica il modello e il tokenizer pre-addestrato
model_id = "sapienzanlp/modello-italia-9b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Carica e tokenizza il dataset
dataset = load_dataset('text', data_files={'train': 'codice_civile_cleaned.txt'})
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)

# Definisci gli argomenti di addestramento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Crea il trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train']
)

# Esegui il fine-tuning
trainer.train()
```

### Validazione del Modello

1. **Dataset di Validazione**: Crea un set di dati separato per validare le prestazioni del modello.
2. **Metriche di Valutazione**: Utilizza precisione, richiamo e F1-score per valutare le prestazioni.

Seguendo questi passaggi, potrai creare un dataset del Codice Civile italiano e utilizzare il modello Italia 9B per generare risposte legali accurate.
