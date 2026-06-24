"""A tool for generating histograms from a CSV file."""

import logging
import os

import pandas
import numpy

import plotly.express
import plotly.subplots
import plotly.graph_objects
import plotly.io

from absl import app
from absl import flags

FLAGS = flags.FLAGS

DEFAULT_SUBPLOT_SECTIONS = [
    'TotalInstructionCount', 'BasicBlockCount', 'TopLevelLoopCount',
    'DirectCallCount', 'LoadInstCount', 'StoreInstCount',
    'IntegerInstructionCount', 'FloatingPointInstructionCount'
]

flags.DEFINE_multi_string('data_path', None, 'The path to the data file.')
flags.DEFINE_string('output_path', None,
                    'The path to a folder to write the histograms to.')
flags.DEFINE_string('data_output_path', None,
                    'The path to a CSV where the data is written to.')
flags.DEFINE_integer('num_bins', 12,
                     'The number of bins to use for the histograms.')
flags.DEFINE_multi_string(
    'sub_plot_sections', DEFAULT_SUBPLOT_SECTIONS,
    'The column names to include in a subplot diagram. There must be eight '
    'sections specified. If this flag is set, only one plot will be generated.')

flags.mark_flag_as_required('data_path')
flags.mark_flag_as_required('output_path')

FANCY_PROPERTY_NAMES = {
    'BasicBlockCount': 'Basic Blocks',
    'TotalInstructionCount': 'Instructions',
    'TopLevelLoopCount': 'Top-level Loops',
    'LoadInstCount': 'Load Instructions',
    'StoreInstCount': 'Store Instructions',
    'DirectCallCount': 'Direct Calls',
    'IntegerInstructionCount': 'Integer Instructions',
    'FloatingPointInstructionCount': 'Floating Point Instructions'
}


def main(_):
  data_frames = []
  languages = []

  for data_path in FLAGS.data_path:
    logging.info(f'Loading data from {data_path}')
    data_frame = pandas.read_csv(data_path, engine='pyarrow')
    data_frame.drop(['name'], axis=1, inplace=True)
    language_name = os.path.basename(data_path)[:-4]
    languages.append(language_name)
    data_frame.insert(0, 'language', [language_name] * len(data_frame))
    data_frames.append(data_frame)

  data_frame = pandas.concat(data_frames)

  logging.info('Finished loading data, generating histograms.')

  if FLAGS.sub_plot_sections is None:
    for column in data_frame:
      figure = plotly.express.histogram(
          data_frame,
          x=column,
          color='language',
          nbins=FLAGS.num_bins,
          log_y=True,
          barmode='overlay')
      figure.write_image(os.path.join(FLAGS.output_path, f'{column}.png'))
      logging.info(f'Finished generating figure for {column}')
    return

  subplot_titles = [
      FANCY_PROPERTY_NAMES[property_key]
      for property_key in FLAGS.sub_plot_sections
  ]

  subplot_figure = plotly.subplots.make_subplots(
      rows=2, cols=4, subplot_titles=subplot_titles)

  output_data_path_handle = None
  if FLAGS.data_output_path:
    output_data_path_handle = open(FLAGS.data_output_path, 'w')

  for index, sub_plot_section in enumerate(FLAGS.sub_plot_sections):
    column = (index % 4) + 1
    row = int(index / 4 + 1)

    for language_index, language in enumerate(languages):
      column_max = data_frame[sub_plot_section].max()
      data_frame_subset = data_frame[data_frame['language'] == language]
      to_show_legend = True if index == 0 else False
      data = data_frame_subset[sub_plot_section].to_numpy()
      counts, bins = numpy.histogram(
          data, bins=FLAGS.num_bins, range=(0, column_max))

      if output_data_path_handle:
        output_data_path_handle.write(f'{language}-{sub_plot_section}\n')

        for current_bin in bins:
          output_data_path_handle.write(f'{current_bin},')
        output_data_path_handle.write('\n')

        for current_count in counts:
          output_data_path_handle.write(f'{current_count},')
        output_data_path_handle.write('\n')

      subplot_figure.add_bar(
          x=bins,
          y=counts,
          name=language,
          row=row,
          col=column,
          marker_color=plotly.colors.qualitative.Plotly[language_index],
          showlegend=to_show_legend)
      subplot_figure.update_layout(barmode='group')
      subplot_figure.update_yaxes(
          type="log", col=column, row=row, exponentformat='power')
      logging.info(
          f'Finished generating figure for {sub_plot_section} in {language}')

  output_data_path_handle.close()

  subplot_figure.update_layout(width=2200, height=1000, font=dict(size=30))
  subplot_figure.update_annotations(font_size=40)

  logging.info('Writing image to file')

  plotly.io.kaleido.scope.mathjax = None

  subplot_figure.write_image(
      os.path.join(FLAGS.output_path, 'subplot_figure.pdf'))


if __name__ == '__main__':
  app.run(main)
