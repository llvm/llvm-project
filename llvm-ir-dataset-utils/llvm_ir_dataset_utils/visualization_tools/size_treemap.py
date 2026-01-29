"""Visualization tool for generating a treemap of size information."""

import os

import pandas

import plotly.express

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'size_file', None,
    'A size file to load data from. Can be specified more than once.')
flags.DEFINE_string('output_file', None,
                    'The output file to place the image in.')
flags.DEFINE_integer(
    'size_threshold', 100 * 1000**2,
    'The size threshold before putting a project in the other category (in bytes).'
)

flags.mark_flag_as_required('size_file')
flags.mark_flag_as_required('output_file')


def load_sizes_file(size_file_path):
  other_size = 0
  total_size = 0
  with open(size_file_path) as size_file:
    name_size_pairs = []
    for line in size_file:
      name_size_pair = line.rstrip().split(',')
      name = name_size_pair[0]
      size = int(name_size_pair[1])
      total_size += size
      if size < FLAGS.size_threshold:
        other_size += size
        continue
      name_size_pairs.append((name, size))
  # Get the basename of the file without the extension
  language_name_base = os.path.basename(size_file_path)[:-4]
  language_name = f'{language_name_base} ({str(round(total_size / 10 ** 9,0))[:-2]} GB)'
  names = [language_name]
  languages = ['ComPile']
  values = [0]
  text = ['']
  for name, size in name_size_pairs:
    size_mb_string = str(round(size / 10**6, 0))[:-2]
    names.append(name + size_mb_string)
    languages.append(language_name)
    values.append(size)
    text.append(f'{size_mb_string} MB')
  other_size_gb = str(round(other_size / 10**9, 2))
  names.append(f'Small {language_name_base} projects')
  text.append(f'Small {language_name_base} projects ({other_size_gb} GB).')
  languages.append(language_name)
  values.append(other_size)
  return (names, languages, values, text)


def main(_):
  names = ['ComPile']
  languages = ['']
  sizes = [0]
  text = ['']

  for size_file in FLAGS.size_file:
    new_names, new_languages, new_sizes, new_text = load_sizes_file(size_file)
    names.extend(new_names)
    languages.extend(new_languages)
    sizes.extend(new_sizes)
    text.extend(new_text)

  data_frame = pandas.DataFrame(
      list(zip(names, languages, sizes)),
      columns=['names', 'languages', 'sizes'])

  figure = plotly.express.treemap(
      data_frame=data_frame,
      names='names',
      parents='languages',
      values='sizes',
      color='sizes',
      color_continuous_scale='Aggrnyl',
      width=1100,
      height=550)

  figure.data[0].text = text
  figure.data[0].textinfo = 'text'

  figure.update_layout(margin=dict(l=20, r=20, t=20, b=20),)

  figure.write_image(FLAGS.output_file)


if __name__ == '__main__':
  app.run(main)
