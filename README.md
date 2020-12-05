# corenlp-client

## prerequisites
[Java 8](https://www.oracle.com/java/technologies/javase-downloads.html)

Python >= 3.6

## Description
A simple, user-friendly python wrapper for Stanford CoreNLP, an nlp tool for natural language processing in Java. 
CoreNLP provides a lingustic annotaion pipeline, which means users can use it to tokenize, ssplit(sentence split), POS, NER, constituency parse, dependency parse, openie etc. However, it's written in Java, which can not be interacted directly with Python programs. Therefore, we've developed a CoreNLP client tool in Python.
The corenlp-client can be used to start a CoreNLP Server once you've followed the official [release](https://stanfordnlp.github.io/CoreNLP/download.html) and download necessary packages and corresponding models. Or, if a server is already started, the only thing you need to do is to specify the server's url, and call the annoate method. 

## Installation
pip install corenlp_client

## Usage

**Quick start**:

Sometimes you may want to get directly tokenized (pos, ner) result in list format without extra efforts. So, we provide `tokenize(), pos_tag(), ner()` methods to simplify the whole process.

```python
from corenlp_client import CoreNLP
# max_mem: max memory use, default is 4. threads: num of threads to use, defualt is num of cpu cores.
annotator = CoreNLP(annotators="tokenize", corenlp_dir="/path/to/corenlp", local_port=9000, max_mem=4, threads=2)
# you can set `clean_text=True` to remove extra spaces in your text
# by setting `ssplit=False`, you'll get a list of tokens without splitting sentences
tokenized_text = annotator.tokenize(data, ssplit=False, clean_text=True)
pos_tags = annotator.pos_tag(data)
ners = annotator.ner(data)
annotator.close()
```

**Start a new server and annotate text**:

If you want to start a server locally, it's more graceful to use `with ... as ...` to handle exceptions.

```python
from corenlp_client import CoreNLP
# max_mem: max memory use, default is 4. threads: num of threads to use, defualt is num of cpu cores.
with CoreNLP(annotators="tokenize", corenlp_dir="/path/to/corenlp", local_port=9000, max_mem=4, threads=2) as annotator:
    # your code here
```

**Use an existing server**: 

You can also use an existing server by providing the url.

```python
from corenlp_request import CoreNLP
# lang for language, default is en.
# you can specify annotators to use by passing `annotator="tokenize,ssplit"` args to CoreNLP. If not provided, all available annotators will be used.
with CoreNLP(url="https://corenlp.run", lang="en") as annotator:
    # your code here
```

**Advanced Usage**

For advanced users, you may want to have access to server's original response in dict format:

```python
anno = annotator.annotate("CoreNLP is your one stop shop for natural language processing in Java! Enjoy yourself! ")
print(anno.tokens) # tokens
print(anno.parse_tree) # parse
print(anno.bi_parse_tree) # binaryParse
print(anno.basic_dep) # basicDependencies
print(anno.enhanced_dep) # enhancedDependencies
print(anno.enhanced_pp_dep) # enhancedPlusPlusDependencies
print(anno.entities) # entitymentions
print(anno.openie) # openie

print(anno.ann_result) # original server's response format
print(anno.pretty_print_tree(anno.parse_tree[0])) # pretty print parse tree's structure
```

**Extra Notes**

- If you choose to start server locally, it'll take a while to load models for the first time you annotate a sentence.

- For timeout error, a simple retry may be useful.

- Also, if "with" is not used, remember to call close() method to stop the Java CoreNLP server. 



