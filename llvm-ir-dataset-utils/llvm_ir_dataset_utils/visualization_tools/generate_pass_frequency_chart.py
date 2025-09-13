"""A tool for generating bar charts describing how often passes run using a
bar chart.
"""

import logging
import os

import pandas
import plotly.graph_objects as go

from absl import app
from absl import flags

from llvm_ir_dataset_utils.util import pass_list_constants

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('data_path', None, 'The path to the data file.')
flags.DEFINE_string('output_file', None,
                    'The path to place the output image at.')
flags.DEFINE_bool('combine_passes', False,
                  'Whether or not to combine passes that run multiple times.')

flags.mark_flag_as_required('data_path')
flags.mark_flag_as_required('output_file')


def main(_):
  bar_charts = []

  full_data_frame = pandas.DataFrame.from_dict({
      'language': [],
      'label': [],
      'percentage': []
  })

  language_names = []

  for language_data_path in FLAGS.data_path:
    language_name = os.path.splitext(os.path.basename(language_data_path))[0]
    language_names.append(language_name)
    logging.info(f'Loading data from {language_data_path}.')
    data_frame = pandas.read_csv(language_data_path)
    data_frame.drop(['name'], axis=1, inplace=True)

    # Only grab passes that we have in case we only have a sample that misses
    # some of the loop passes. We're only transforming everything to make sure
    # it's all in order.
    passes_to_grab = []
    for pass_name in pass_list_constants.OPT_DEFAULT_O3_PASS_LIST:
      if pass_name in data_frame.columns:
        passes_to_grab.append(pass_name)

    data_frame = data_frame[passes_to_grab]

    labels = []
    percentages = []

    for column in data_frame.keys():
      percentage = data_frame[column].sum() / data_frame.shape[0]
      if percentage != 0:
        pass_name = column.split('Pass')[0]
        if not FLAGS.combine_passes:
          pass_name += column[-1]
        labels.append(pass_name)
        percentages.append(data_frame[column].sum() / data_frame.shape[0])

    partial_data_frame = pandas.DataFrame.from_dict({
        'language': [language_name for _ in range(0, len(labels))],
        'label': labels,
        'percentage': percentages
    })

    full_data_frame = pandas.concat([full_data_frame, partial_data_frame])

    logging.info(
        f'Finished generating data for plot with {len(labels)} labels for {language_name}'
    )

  logging.info('Finished loading data, generating figures.')

  full_data_frame.sort_values(by=['percentage'], ascending=False, inplace=True)

  for language_name in language_names:
    data_frame = full_data_frame[full_data_frame['language'] == language_name]
    bar_charts.append(
        go.Bar(
            name=language_name,
            x=data_frame['percentage'],
            y=data_frame['label'],
            orientation='h'))

  figure = go.Figure(data=bar_charts)

  figure.update_layout(
      barmode='group', height=1500, width=1000, font=dict(size=20))
  figure.update_xaxes(type="log", exponentformat='power')
  figure.update_yaxes(autorange="reversed")

  figure.write_image(FLAGS.output_file)


if __name__ == '__main__':
  app.run(main)
