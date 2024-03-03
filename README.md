# AI-Powered-Text-Classifier-Harnessing-Large-Language-Models-for-Precise-Data-Categorization(Prompting Techniques/RAG)
## Problem Statement
### Dataset: train 40k .csv
Design and implement a classifier using any LLM to classify the data in Column name “Text” with Column name “Cat2” and Column name “Cat3”.
- Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 2
- Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 3

Report the Accuracy on a sample test set split from 40k samples.

## SOLUTION
### DATASET PREPARATION
In addressing this classification problem and aiming to construct a classifier using a Large Language Model (LLM), the data must be formatted in a specific manner for fine-tuning. In this case, I am opting to prepare the data in Alpaca format.The dataset has been segregated into inference data and training data.The initial 1000 datapoints are designated as inference data, while the remaining datapoints are allocated for training data.

### Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 2
Following steps are performed for data preparation.
- Taking column Text and Cat2 from the dataset.
- Creating a column called Instruction and add the prompt :**Classify the text into categories given in output column. Reply with only the words given in output column.**
- Rename the columns Text and Cat2 as input and output.
- Finally create a column called text and add the content as below:
```python
### InstructionInstruction.### Input:input+### Output:output
```
For more details ,please refer the notebook :**training data preparation2 .ipynb**
### Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 3
Following steps are performed for data preparation.
- Taking column Text and Cat2 from the dataset.
- Creating a column called Instruction and add the prompt :**Classify the text into categories given in output column. Reply with only the words given in output column.**
- Rename the columns Text and Cat3 as input and output.
- Finally create a column called text and add the content as below:
```python
    ### InstructionInstruction.### Input:input+### Output:output
```
For more details ,please refer the notebook :**training data preparation2 .ipynb**
## MODEL FINE TUNING
For our task, we’re employing the Llama 2 model. Given that Llama 2 lacks specific knowledge about our data domain, we plan to enhance its performance by fine-tuning the model. This approach aims to yield improved results tailored to our specific domain.
### Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 2
Following steps are used for fine tuning the Model:
- Load the data which is prepared from data preparation step.
- Load the model Llama 2 from hugging face(Note: We require Access tokens from hugging face to access this model)
- Load the tokenizer
- We are going to use PEFT technique for fine tuning the model to save the computation and cost.
- Finally training has been started .
- I have pushed the fine tuned model to hugging face hub: [Model](https://huggingface.co/Sakil/llama2-fine-tuned-classfier-cat2 "Model")

### Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 3
- Load the data which is prepared from data preparation step.
- Load the model Llama 2 from hugging face(Note: We require Access tokens from hugging face to access this model)
- Load the tokenizer
- We are going to use PEFT technique for fine tuning the model to save the computation and cost.
- Finally training has been started .
- I have pushed the fine tuned model to hugging face hub : [Model](https://huggingface.co/Sakil/llama2-fine-tuned-classfier-cat3 "Model")
