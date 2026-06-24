"""This script loads in a folder of parquet files from the process_to_parquet.py
script and validates some of the fields. This is not an exhaustive validation
and only contains simple smoke tests, such as ensuring fields are not empty.
"""

import logging
import os

from absl import app
from absl import flags

from pyarrow import parquet

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', None, 'The dataset path to validate')


def main(_):
  total_rows = 0

  for file_name in os.listdir(FLAGS.dataset_path):
    full_file_path = os.path.join(FLAGS.dataset_path, file_name)

    # Load the parquet file
    current_table = parquet.read_table(
        full_file_path,
        columns=[
            'license_expression', 'license_source', 'license_files',
            'package_source', 'language'
        ]).to_pandas()

    warning_count = 0

    for index, module_instance in current_table.iterrows():
      total_rows += 1
      if len(module_instance['license_expression']) == 0:
        warning_count += 1
        logging.info('License expression empty')
      if len(module_instance['license_source']) == 0:
        warning_count += 1
        logging.info('License source empty')
      if len(module_instance['license_files']) == 0:
        warning_count += 1
        logging.info('License files empty')
      if len(module_instance['package_source']) == 0:
        warning_count += 1
        logging.info('Package source empty')
      if len(module_instance['language']) == 0:
        warning_count += 1
        logging.info('Language field empty')

    logging.info('Finished processing individual dataset file.')

  logging.info(f'Finished processing dataset, found {total_rows} rows.')


if __name__ == '__main__':
  app.run(main)
