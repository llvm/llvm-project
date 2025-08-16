"""A tool for counting various quantities like tokens from gathered statistics
CSV files.
"""

import logging
import csv

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'stats_path', None, 'The path to a statistics file containing some count.')
flags.DEFINE_string('key', 'token_count', 'The column in the CSV to sum over.')


def count_values_from_file(file_path):
  count = 0
  with open(file_path) as count_file:
    count_reader = csv.DictReader(count_file)
    for count_entry in count_reader:
      count += int(count_entry[FLAGS.key])
  return count


def main(_):
  total_count = 0
  for stats_path in FLAGS.stats_path:
    total_count += count_values_from_file(stats_path)

  logging.info(f'Total for column {FLAGS.key} {total_count}.')


if __name__ == '__main__':
  app.run(main)
