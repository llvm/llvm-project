"""Tool for generating a histogram of external functions that get called."""

import logging
import os
import csv

import pandas
import plotly.express

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'call_data_path', None,
    'A path to a file containing a list of function calls.')
flags.DEFINE_multi_string(
    'defined_functions_path', None,
    'A path to a file containing a list of defined functions.')
flags.DEFINE_string('output_file', None, 'The path to the output image.')

flags.mark_flag_as_required('call_data_path')
flags.mark_flag_as_required('defined_functions_path')
flags.mark_flag_as_required('output_file')


def get_definitions_per_project(file_path):
  project_functions = {}
  with open(file_path) as definitions_file:
    definition_reader = csv.DictReader(definitions_file)
    for definition in definition_reader:
      project_path = definition['name'].split(':')[0]
      if project_path in project_functions:
        project_functions[project_path].add(definition['defined_function'])
      else:
        project_functions[project_path] = set([definition['defined_function']])
  return project_functions


def load_external_calls(file_path, project_functions):
  external_calls = []
  with open(file_path) as calls_file:
    call_reader = csv.DictReader(calls_file)
    for function_call in call_reader:
      project_path = function_call['name'].split(':')[0]
      called_function = function_call['call_names']
      if called_function in project_functions[project_path]:
        continue
      external_calls.append(called_function)
  return external_calls


def generate_calls_histogram(external_calls):
  call_histogram = {}
  for external_call in external_calls:
    if external_call in call_histogram:
      call_histogram[external_call] += 1
    else:
      call_histogram[external_call] = 1
  return call_histogram


def main(_):
  project_functions = {}

  for defined_functions_path in FLAGS.defined_functions_path:
    project_functions.update(
        get_definitions_per_project(defined_functions_path))

  external_calls = []

  for call_data_path in FLAGS.call_data_path:
    external_calls.extend(
        load_external_calls(call_data_path, project_functions))

  external_call_histogram = generate_calls_histogram(external_calls)

  external_call_names = []
  external_call_frequencies = []

  for external_call in external_call_histogram:
    external_call_names.append(external_call)
    external_call_frequencies.append(external_call_histogram[external_call])

  data_frame = pandas.DataFrame({
      'call_name': external_call_names,
      'count': external_call_frequencies
  })

  data_frame.sort_values(by=['count'], inplace=True, ascending=False)

  figure = plotly.express.bar(data_frame.head(20), x='call_name', y='count')

  figure.write_image(FLAGS.output_file)


if __name__ == '__main__':
  app.run(main)
