"""A tool for generating visualizations of bitcode distributions across
languages.
"""

import logging
import os
import csv

import pandas
import plotly.express
import plotly.io

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('bc_dist_file', None, 'The path to a data file.')
flags.DEFINE_multi_string(
    'opt_bc_dist_file', [],
    'The path to a data file containing data gathered post-optimization.')
flags.DEFINE_string('output_file', None, 'The path to the output image.')
flags.DEFINE_string(
    'output_data_file', None,
    'The path to a CSV file to place the data used to generate the figure.')

flags.mark_flag_as_required('bc_dist_file')
flags.mark_flag_as_required('output_file')

OPCODES_TO_ANALYZE = [
    'Load', 'GetElementPtr', 'Call', 'BitCast', 'Store', 'Alloca', 'Br',
    'AddrSpaceCast', 'Ret', 'ICmp', 'ExtractValue', 'Invoke'
]


def compute_cumulative_histogram_from_file(file_path):
  histogram = {}
  with open(file_path) as bc_dist_file:
    dict_reader = csv.DictReader(bc_dist_file)
    for data_row in dict_reader:
      for instruction_type in data_row:
        if instruction_type == 'name':
          continue
        instruction_count = int(data_row[instruction_type])
        if instruction_type in histogram:
          histogram[instruction_type] += instruction_count
        else:
          histogram[instruction_type] = instruction_count
  return histogram


def main(_):
  distributions = {}
  instruction_names = []
  for bc_dist_file_path in FLAGS.bc_dist_file + FLAGS.opt_bc_dist_file:
    logging.info(f'Loading data from {bc_dist_file_path}')
    language_name = os.path.basename(bc_dist_file_path)[:-4]
    if bc_dist_file_path in FLAGS.opt_bc_dist_file:
      language_name += ' (Optimized)'
    distribution = compute_cumulative_histogram_from_file(bc_dist_file_path)
    instruction_names = list(set(instruction_names + list(distribution.keys())))
    distributions[language_name] = distribution

  # Ensure that all languages have the same opcodes.
  for distribution in distributions:
    for instruction_name in instruction_names:
      if instruction_name not in distributions[distribution]:
        distributions[distribution][instruction_name] = 0

  # Normalize the distributions in each language by the instruction count
  for distribution in distributions:
    total_instruction_count = 0
    for instruction_name in distributions[distribution]:
      total_instruction_count += distributions[distribution][instruction_name]
    for instruction_name in distributions[distribution]:
      distributions[distribution][instruction_name] = distributions[
          distribution][instruction_name] / total_instruction_count

  # Remove all opcodes that aren't in the set that we want.
  for distribution in distributions:
    for instruction_name in instruction_names:
      if instruction_name not in OPCODES_TO_ANALYZE:
        del distributions[distribution][instruction_name]

  # Add an additional opcodes category so everything sums to one.
  extra_percentages = {}
  for distribution in distributions:
    total_instruction_count = 0
    for instruction_name in distributions[distribution]:
      total_instruction_count += distributions[distribution][instruction_name]
    extra_percentages[distribution] = 1 - total_instruction_count

  data_frame = pandas.DataFrame({
      'Language': [],
      'Instruction': [],
      'Count': []
  })

  for language_name in distributions:
    language_names = []
    instructions = []
    instruction_counts = []

    for instruction in distributions[language_name]:
      language_names.append(language_name)
      instructions.append(instruction)
      instruction_counts.append(distributions[language_name][instruction])

    language_data_frame = pandas.DataFrame({
        'Language': language_names,
        'Instruction': instructions,
        'Count': instruction_counts
    })

    language_data_frame.sort_values('Count', ascending=False, inplace=True)

    data_frame = pandas.concat([data_frame, language_data_frame])

  extra_percentages_df = pandas.DataFrame({
      'Language': list(extra_percentages.keys()),
      'Instruction': [
          'Other Instructions' for _ in range(0, len(extra_percentages))
      ],
      'Count': list(extra_percentages.values())
  })

  data_frame = pandas.concat([data_frame, extra_percentages_df],
                             ignore_index=True)

  if FLAGS.output_data_file:
    data_frame.pivot(
        index='Language', columns='Instruction', values='Count').to_csv(
            FLAGS.output_data_file, index_label='index')

  logging.info('Generating figure.')

  figure = plotly.express.bar(
      data_frame,
      x='Language',
      y='Count',
      color='Instruction',
      color_discrete_sequence=plotly.express.colors.qualitative.Alphabet_r)

  figure.update_layout(legend_traceorder="reversed")

  logging.info('Writing figure to file.')

  plotly.io.kaleido.scope.mathjax = None

  figure.write_image(FLAGS.output_file)


if __name__ == '__main__':
  app.run(main)
