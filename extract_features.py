'''
Create a list of tables with different features such as:
 - number of columns
 - number of rows
 - number of words per cell
 - contains a specific word
 - ...

A column 'is_table' is computed from the features to predict if a table is an actual table.
Majority of extracted tables are not actual and table, but simple text, hence the importance to filter them out.

The input is a directory containing the tables extracted from pdf documents using the 'extract-tables.jar' command line tool.
'''
import pandas as pd
import glob
import os
import re
import collections
import tqdm
import logging

def get_table_parts(file):
    items = file.split(os.sep)
    article = items[-3]
    _, page, _, table_num, _ = items[-1].split('.')
    return (article, int(page), int(table_num))

def get_table_file(dir_, article, page, table):
    return os.path.join(dir_, article, 'tables', 'page.{:03d}.table.{:02d}.csv'.format(page, table))

def read_csv(file):
    return pd.read_csv(file, encoding='utf-8', header=None, dtype=str).fillna('')

def is_word_in(word, cells, case_sensitive, match_end):
    pattern = r'\b{}{}'.format(word, r'\b' if match_end else '')
    flags = re.IGNORECASE if not case_sensitive else 0
    pattern = re.compile(pattern, flags)
    for c in cells:
        for _ in pattern.finditer(c):
            return True
    return False

def predict_is_table(s):
    return s['digits_perc'] > 0.1 and s['cell_words_mean'] < 5

def get_table_features(df):
    cells = [c for _, col in df.items() for c in col]
    words_count_by_cell = pd.Series([len(c.split()) for c in cells])
    features = collections.OrderedDict(
        columns = len(df.columns),
        rows = len(df),
        cell_words_min = words_count_by_cell.min(),
        cell_words_mean = words_count_by_cell.mean(),
        cell_words_max = words_count_by_cell.max(),
        empty_cells = sum(1 for c in cells if c.strip() == ''),
        total_words = sum(1 for c in cells for w in c.split()),
        total_chars = sum(1 for c in cells for char in c),
        total_digits = sum(1 for c in cells for char in c if char in '0123456789')
    )
    features['digits_perc'] = features['total_digits'] / features['total_chars']
    # words features
    words = [
        ('age', False, True),
        ('date', False, True),
        ('year', False, False),
        ('BP', True, True),
        ('BC', True, True),
        ('AD', False, True),
        ('Ka', True, True),
        ('CAL', False, True),
        ('painting', False, False),
        ('drawing', False, False),
        ('engraving', False, False),
        ('pictograph', False, False),
        ('petroglyph', False, False),
        ('AMS', True, True),
        (r'Uranium\sserie', False, False),
        ('Radiocarbon', False, True),
        ('RC14', False, True),
        ('charcoal', False, True),
        ('pigment', False, False),
        ('calcite', False, False),
        ('beeswax', False, True),
        ('varnish', False, True),
        ('bone', False, False),
        ('cave', False, False),
        ('site', False, False),
    ]
    for word, case_sensitive, match_end in words:
        word_name = word if case_sensitive else word.lower()
        word_name = re.sub(r'\\.', '_', word_name)
        feature_name = 'w_{}_{}_{}'.format('cs' if case_sensitive else 'ci', 'e' if match_end else 'ne', word_name)
        features[feature_name] = is_word_in(word, cells, case_sensitive, match_end)
    # digit featues
    digit_word_len_count = collections.Counter(len(d) for c in cells for d in re.findall(r'\d+', c))
    for length in range(2, 5 + 1):
        feature_name = 'digit_of_len_{}'.format(length)
        features[feature_name] = digit_word_len_count.get(length, 0)
    # is an actual table
    features['is_table'] = predict_is_table(features)
    features.move_to_end('is_table', last=False)
    return features

# params
pdf_extraction_dir = r'<input dir>'
output_dir = 'output'

def output(name):
    f = os.path.join(output_dir, name)
    os.makedirs(os.path.split(f)[0], exist_ok=True)
    return f

logging.basicConfig(filename=output('extract_features.log'), level=logging.INFO)

files = glob.glob(os.path.join(pdf_extraction_dir, '*', 'tables', '*.table.*.csv'))
tables = pd.DataFrame([get_table_parts(f) for f in files], columns=['article', 'page', 'table'])

# extract features
feature_list = []
for table in tqdm.tqdm(tables.itertuples(), desc='extracting features', total=len(tables)):
    try:
        file = get_table_file(pdf_extraction_dir, table.article, table.page, table.table)
        df = read_csv(file)
        features = get_table_features(df)
        feature_list.append(features)
    except:
        logging.exception('Failed to extract features for table {} p.{} t.{}'.format(table.article, table.page, table.table))

features_df = pd.DataFrame(feature_list)
tables = pd.concat([tables, features_df], axis=1)

tables.to_pickle(output('tables.pkl'))
tables.to_csv(output('tables.csv'))
