"""Tool for running llvm-link over all bitcode files in a corpus."""

import pathlib
import os
import subprocess
import logging

from absl import app
from absl import flags

import ray

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None, 'The path to the corpus directory.')
flags.DEFINE_string('output_dir', None, 'The path to the output directory.')

flags.mark_flag_as_required('corpus_dir')


@ray.remote(num_cpus=1)
def link_package(folder_path, output_dir):
  # TODO(boomanaiden154): Pull from a corpus_manifest/meta corpus manifest
  # rather than glob for the bitcode files once they're available in all of
  # my builds.
  bitcode_files_gen = pathlib.Path(folder_path).glob('**/*.bc')
  bitcode_files = list(bitcode_files_gen)

  if len(bitcode_files) == 0:
    return (False, None)

  command_vector = ['llvm-link']

  command_vector.append(bitcode_files[0])
  for bitcode_file in bitcode_files[1:]:
    command_vector.extend(['-override', bitcode_file])

  package_name = os.path.basename(folder_path)
  output_file_path = os.path.join(output_dir, package_name + '.bc')
  command_vector.extend(['-o', output_file_path])

  try:
    command_output = subprocess.run(
        command_vector, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  except OSError:
    return (False, None)

  if command_output.returncode == 0:
    return (True, output_file_path)
  else:
    return (False, output_file_path)


def main(_):
  pathlib.Path(FLAGS.output_dir).mkdir(exist_ok=True, parents=True)

  corpus_folders = os.listdir(FLAGS.corpus_dir)

  package_processing_futures = []
  for corpus_folder in corpus_folders:
    corpus_folder_full_path = os.path.join(FLAGS.corpus_dir, corpus_folder)
    package_processing_future = link_package.remote(corpus_folder_full_path,
                                                    FLAGS.output_dir)
    package_processing_futures.append(package_processing_future)

  link_success = 0
  link_failures = []
  while len(package_processing_futures) > 0:
    to_wait_for = 128
    if len(package_processing_futures) < 256:
      to_wait_for = 1
    finished, package_processing_futures = ray.wait(
        package_processing_futures, timeout=5.0, num_returns=to_wait_for)
    finished_data = ray.get(finished)
    for finished_link in finished_data:
      if finished_link[0]:
        link_success += 1
      else:
        link_failures.append(finished_link[1])
    logging.info(
        f'Just finished {len(finished_data)}, {len(package_processing_futures)} remaining.'
    )

  logging.info(
      f'Got {link_success} successes and {len(link_failures)} failures.')


if __name__ == '__main__':
  app.run(main)
