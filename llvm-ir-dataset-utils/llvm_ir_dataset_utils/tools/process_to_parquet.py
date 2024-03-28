"""This is a script that allows for the conversion of a deduplicated dataset
into a parquet dataset for distribution.
"""

import logging
import os
import sys

from absl import app
from absl import flags

import pandas

import pyarrow

from pyarrow import parquet

from llvm_ir_dataset_utils.util import dataset_corpus

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None, 'The corpus to pull bitcode from.')
flags.DEFINE_integer('max_projects', sys.maxsize,
                     'The maximum number of projects to process')

flags.mark_flag_as_required('corpus_dir')

# TODO(boomanaiden154): Add in support for propogating license information
# and other project provenance information once we have it.


def process_single_project(project_dir, dataset_dir):
  try:
    bitcode_paths = dataset_corpus.get_bitcode_file_paths(project_dir)
  except:
    return

  module_content = []

  for bitcode_path in bitcode_paths:
    bitcode_file_data = dataset_corpus.load_file_from_corpus(
        project_dir, bitcode_path)
    module_content.append(bitcode_file_data)

  dataframe = pandas.DataFrame.from_dict({'content': module_content})

  table = pyarrow.Table.from_pandas(dataframe, preserve_index=False)

  parquet.write_table(table, dataset_dir)


def main(_):
  projects_list = os.listdir(FLAGS.corpus_dir)

  logging.info(f'Processing {len(projects_list)} projects')

  for index, project_dir in enumerate(projects_list):
    project_path = os.path.join(FLAGS.corpus_dir, project_dir)
    process_single_project(project_path, '/tmp/test.parquet')
    logging.info(f'Just finished processing {project_dir}')

    if index >= FLAGS.max_projects:
      break


if __name__ == '__main__':
  app.run(main)
