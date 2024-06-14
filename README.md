# LLM-fine-tuning


For this project, I fine-tuned two separate models for three tasks: Document Summarization, Conversation Summarization, and Sentiment Analysis. The code has been separated into three Jupyter notebooks, defined in each summary below. 
To execute the code in these notebooks you will need the following;
+ Access to GPU/TPU.  These notebooks were developed in Google Colab Notebooks
+ A free Hugging Face Hub Account. This is required to download the datasets.  The data is free, but requires you to sign a terms of use agreement

## Model Selection
I considered a number of models to be used as a base model for fine-tuning.  The key selection criteria for this project were as follows:
+ **Model Size:** I would be using Google Colab for fine-tuning with access to limited GPU/TPU.  Smaller models were preferred to reduce train time and compute resources
+ **Easily Accessible:** The model needs to be open source and available on an ungated platform such as HuggingFace
+ **Documentation:**  The model needed to have ample documentation and examples available to guide me through the process
+ **Capability:** The models must be appropriate for text summarization or text classification
+ **Performance:**  The model should have good performance on benchmarks such as MMLU
  
Essentially, a small, well-documented, open-source model with good performance that can be used for text summarization and classification. Google's FLAN-T5 base model and DistilBERT models met all the criteria. 

### FLAN-T5
The [FLAN-T5](https://research.google/blog/introducing-flan-more-generalizable-language-models-with-instruction-fine-tuning/) base model was selected because it is a small yet powerful encoder-decoder model developed by Google, fine-tuned on various tasks. It offers performance [comparable](https://exemplary.ai/blog/flan-t5) to larger language models, is easy to implement and fine-tune with many examples available, and is ideal for learning. Its use cases include text classification, summarization, translation, and question-answering.

### DistilBERT
The [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) model was selected because it is a distilled version of the well-known BERT model by Google, which is renowned for its effectiveness in various NLP tasks. DistilBERT offers a more efficient alternative with 40% fewer parameters and a 60% increase in speed while preserving 95% of BERT's performance. This makes it well-suited for text classification tasks.

## Fine-Tuning
Parameter Efficient Fine Tuning (PEFT) was implemented to fine-tune the models on specific tasks.  Specifically,  the LoRA (Low-Rank Adaptation) technique, as implemented in the HuggingFace library, was selected for the following reasons:
LoRA adds trainable adaptors into specific layers of the model, which capture task-specific information without altering the core parameters of the pre-trained model.
+ This is the preferred method of many practitioners
+ It is effective at improving performance for task-specific fine-tuning
+ It uses much fewer resources than full fine-tuning
+ It only trains a small fraction of model weights (~1%)
+ It prevents catastrophic forgetting because the base model weights are unchanged. 
+ The adaptors are merged with the original base model weights
+ There is only a small loss of performance when compared to full fine-tuning

## Hyperparameter Tuning 
To optimize hyperparameters, I normally use grid-search, random-search, or use a Bayesian optimization method. Grid search is thorough but time-consuming, while random-search/Bayesian methods can be faster but may not find globally optimal results.  Due to time and resource constraints, I didn’t perform a systematic hyperparameter search for this project. Instead, rudimentary trial-and-error experiments were conducted to improve model performance using a small number of epochs.  As a result, the fine-tuned models are likely not optimal and could be improved by focused hyperparameter tuning.

## Model Evaluation
### Sentiment Analysis/ Classification
The IMDB dataset provides a movie review and a human-assigned classification (positive, negative) of that review. This is a supervised binary classification task (we have the ground truth labels). Therefore, a classification accuracy measure can be used. The F1 score was selected to balance precision and recall;

To evaluate the document and dialogue summaries, the following steps were taken;
+ Randomly select 500 examples from the test dataset (out of sample)
+ Compare the predictions to the human label for each example using the original base model and the fine-tuned model
+ Calculate the overall F1 score for all 500 examples for each model
+ The fine-tuned LLM improved the model's ability to classify movie reviews when assessed using an F1 accuracy score compared to the base model (89% vs 66%)

### Text Summarization
Evaluating model performance for text summarization is much more complex than regression or classification. Qualitative assessments can be done by comparing individual summaries, but this technique is not practical for evaluating large datasets and does not provide a non-subjective method of comparison. A method to quantitatively compare summaries is required, and fortunately, several methods are available on the Hugging Face platform. The following metrics were selected to evaluate the document and dialogue summarization:

+ The [Rouge Score](https://huggingface.co/spaces/evaluate-metric/rouge) (recall-oriented understudy of gisting evaluation) compares the overlap of n-grams between the generated and human summaries. The score ranges from 0 (no overlap) to 1 (perfect overlap). For this project, rouge1, rouge2, and rougeL (longest common sequence) were used.
+ The [BERT Score](https://huggingface.co/spaces/evaluate-metric/bertscore) calculates the semantic similarity between the generated and human summaries. The values range from 0 (not similar) to 1 (perfectly similar ).  For this project, the F1 BERT score was used.

To evaluate the document and dialogue summaries, the following steps were taken;
+ Randomly select 50 examples from the test dataset (out of sample)
+ Generate summaries for each example from the base and tuned models
+ Calculate the evaluation metrics 
+ Plot the metrics in BoxPlots to compare the median values and distribution of values

  
![dialogue](https://github.com/kconstable/LLM-fine-tuning/assets/1649676/42447788-e601-4842-9c42-25fce3d39251)


![text-summraization](https://github.com/kconstable/LLM-fine-tuning/assets/1649676/e9ed98b9-67fc-4d7c-b71f-3e82a4bb0e1e)
