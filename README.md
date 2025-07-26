# PyPremise

Premise is a data mining method to identify significant differences between two groups of texts. It can be used, e.g., to identify from the outputs of an LLM what systematic differences occur if you change your prompt. Or it can identify patterns or explanations of where an NLP classifier performs well 
and where it fails.

PyPremise is a Python library to makes it easy to run Premise with just a couple of lines of code.

## Example Applications

### Systematic differences between two prompts or two LLMs
Premise identified, e.g., that changing the last name of the protagonist from Dr. Li to Dr. Smith in a story generation prompt caused - among other differences - a drastic change in the gender distribution (more details and examples [here](https://aclanthology.org/2025.acl-long.985/))

| Patterns Group 0 (Dr. Li)                                      | Patterns Group 1 (Dr. Smith)       |
|----------------------------------------------|--------------------------------|
| (She, she)                                  | (He, he)   |
| (car, accident)             | (general, practioner)        |
|                                | (small, town) |

### Evaluating where a classifier fails
Premise identified, e.g., for a Visual Question Answering model that it struggled with counting, color identification and visual orientation (more details and examples [here](https://proceedings.mlr.press/v162/hedderich22a.html)):

| pattern                                      | example from the dataset       |
|----------------------------------------------|--------------------------------|
| (how, many)                                  | how many elephants are there   |
| (what, ⓧ(color, colors, colour) )            | what color is the bench        |
| (on, top, of)                                | what is on the top of the cake |
| (left, to)                                   | what can be seen to the left   |
| (on, wall, hanging)                          | what is hanging on the wall    |




## Example Usage
You can run
PyPremise with just a few lines of code.

```python
from pypremise import Premise, data_loaders
premise_instances, _, voc_index_to_token = data_loaders.get_dummy_data()

premise = Premise(voc_index_to_token=voc_index_to_token)
patterns = premise.find_patterns(premise_instances)

for p in patterns:
    print(p)
# prints two patterns
# (How) and (many) towards group 0
# (When) and (was) and (taken) towards group 
```

PyPremise provides you with helper methods to load data from different sources like numpy arrays or tokenized text files. See below for more examples.

## Installation
Install a recent version of Python, then just run
```
pip install pypremise
```

Currently, PyPremise directly supports Linux (Ubuntu) and Mac (Apple Silicon, the "M" processors). We are currently working on a Windows port.

## Documentation
- General introductory example: [`general_example.ipynb`](./documentation/general_example.ipynb)
- How to evaluate LLM outputs: [`LLM_outputs_examples.ipynb`](./documentation/LLM_outputs_examples.ipynb)
- How to explain misclassifications: [`missclassification_examples.ipynb`](./documentation/missclassification_examples.ipynb)
- How to use word embeddings for more complex results: [`word_embedding_examples.ipynb`](./documentation/word_embedding_examples.ipynb)

## Code Reference

The file [`data_loaders.py`](./src/pypremise/data_loaders.py) contains helper functions for loading data from various sources such as token lists, NumPy-like arrays, or CSV files. If you want to load your own data, refer to this file for examples and supported formats.

## Issues, License & Citation

If you run into any issues using PyPremise, do not hesitate to contact us (e-mail in the publication) or create an issue on Github.

If you use this tool in your work, we would be happy if you tell us about it.

If you use PyPremise to evaluate the output of LLMs, please cite

```
@inproceedings{hedderich-etal-2025-whats,
    title = "What{'}s the Difference? Supporting Users in Identifying the Effects of Prompt and Model Changes Through Token Patterns",
    author = "Hedderich, Michael A.  and
      Wang, Anyi  and
      Zhao, Raoyuan  and
      Eichin, Florian  and
      Fischer, Jonas  and
      Plank, Barbara",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.985/"
}
```

For any other uses (like e.g. explaining misclassifications) and for the general Premise algorithm, please cite:

```
@inproceedings{premise,
  author    = {Michael A. Hedderich and
               Jonas Fischer and
               Dietrich Klakow and
               Jilles Vreeken},
  title     = {Label-Descriptive Patterns and Their Application to Characterizing
               Classification Errors},
  booktitle = {International Conference on Machine Learning, {ICML}},
  series    = {Proceedings of Machine Learning Research},
  year      = {2022},
  url       = {https://proceedings.mlr.press/v162/hedderich22a.html}
}
```

PyPremise (not Premise itself) is published under MIT license.
