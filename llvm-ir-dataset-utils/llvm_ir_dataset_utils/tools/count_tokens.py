"""A tool for counting tokens from gathered statistics CSV files."""

import logging
import csv

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'stats_path', None,
    'The path to a statistics file containing a token count.')
flags.DEFINE_string('key', 'token_count', 'The column in the CSV to sum over.')


def count_tokens_from_file(file_path):
  token_count = 0
  with open(file_path) as token_count_file:
    token_count_reader = csv.DictReader(token_count_file)
    for token_count_entry in token_count_reader:
      token_count += int(token_count_entry[FLAGS.key])
  return token_count


def main(_):
  total_token_count = 0
  for stats_path in FLAGS.stats_path:
    total_token_count += count_tokens_from_file(stats_path)

  logging.info(f'Counted {total_token_count} tokens.')


if __name__ == '__main__':
  app.run(main)
