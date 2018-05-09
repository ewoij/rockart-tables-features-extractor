# Rock Art Tables Features Extractor

Generates data frames indexing tables extracted using [pdf-tables-extractor](https://github.com/ewoij/pdf-tables-extractor).

The script will predict using a simple heuristic if a table is a real table and will also create some boolean columns indicating the presence of specific words.

## Requirements

Python 3.6
 - tqdm>=4.19.5
 - pandas>=0.20.3

## Run

Update the following variables with the input/output directories:

```python
pdf_extraction_dir = r'<input dir>'
output_dir = 'output'
```