"""Tool for getting the top x constants from a constant frequency histogram."""

import logging

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('constant_histogram_file', None,
                    'The path to the constant histogram CSV file.')
flags.DEFINE_string('output_file', None, 'The path to the output file.')
flags.DEFINE_integer('constant_count', None,
                     'The number of constants to pull from the histogram.')

flags.mark_flag_as_required('constant_histogram_file')
flags.mark_flag_as_required('output_file')
flags.mark_flag_as_required('constant_count')


def main(_):
  constants = []
  with open(FLAGS.constant_histogram_file) as constant_histogram_file:
    for line in constant_histogram_file:
      line_stripped = line.rstrip()
      line_parts = line_stripped.split(',')
      constants.append((int(line_parts[0]), int(line_parts[1])))

  constants.sort(key=lambda const: const[1], reverse=True)

  with open(FLAGS.output_file, 'w') as output_file:
    for constant in constants[0:FLAGS.constant_count]:
      output_file.write(f'{constant[0]}\n')


if __name__ == '__main__':
  app.run(main)
