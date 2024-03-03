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

## EXPERIMENT WITH FINE TUNED MODEL AND PROMPTING TECHNIQUES
We utilized [Langchain](https://python.langchain.com/docs/get_started/introduction "Langchain") along with our fine-tuned model to build the classifier.
### Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 2
Following steps are performed to build the system:
- Load the inference data which was done in data preparation step
- Import PromptTemplate, LLMChain from Langchain
- Create the list of unique categories from the column Cat 2 from the inference data.
- Develop a prompt
- I have followed few short prompting for the purpose our task.
- For more details : Please refer the Notebook : **training cat2 llama2.ipynb**
### Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 3
Following steps are performed to build the system:
- Load the inference data which was done in data preparation step
- Import PromptTemplate, LLMChain from Langchain
- Create the list of unique categories from the column Cat 3 from the inference data.
- Develop a prompt
- I have followed few short prompting for the purpose our task.
- For more details : Please refer the Notebook : **training cat3 llama2.ipynb**
## RESULT & METRICS
### Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 2
For simplicity, I have selected 100 data points for inference, and within this sample, there are 24 categories.

![](https://github.com/Sakil786/AI-Powered-Text-Classifier-Harnessing-Large-Language-Models-for-Precise-Data-Categorization/blob/main/image1.png)

Out of 100 records , the model is able to predict 46 records correctly .
Note : The inference result can be found in csv file entitled : **inference data cat2 with accuracy.csv**
### Input to your prompt will be a text from column name Text, and output should be class name from Column Name Cat 3
For simplicity, I have selected 100 data points for inference, and within this sample, there are 34 categories.
![](https://github.com/Sakil786/AI-Powered-Text-Classifier-Harnessing-Large-Language-Models-for-Precise-Data-Categorization/blob/main/image2.png)

Out of 100 records , the model is able to predict 14 records correctly .
Note : The inference result can be found in csv file entitled :** inference data cat3 with accuracy.csv**
## IMPROVEMENT SUGGESTIONS
- The model is demonstrating satisfactory performance for the category Cat2 .
- Enhancing the results is possible by incorporating diverse sample categories and experimenting with various prompts.
- Fine-tuning with specific prompts has the potential to enhance the results for cat3.
- Implementing the RAG technique could potentially enhance the outcomes for both cat2 and cat3.
## Explore, Appreciate, and Give the Repository a Shining ⭐
Feel free to explore the repository and show your appreciation by giving it a star⭐! Your support means a lot! 😉

