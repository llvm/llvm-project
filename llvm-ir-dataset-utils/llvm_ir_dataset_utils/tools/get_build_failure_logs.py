"""Tool to find all the logs for targets that failed to build from a corpus
directory."""

import glob
import os
import json
import logging

from absl import app
from absl import flags

import ray

from llvm_ir_dataset_utils.util import dataset_corpus

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None,
                    'The corpus directory to look for build logs in.')

flags.mark_flag_as_required('corpus_dir')


@ray.remote(num_cpus=1)
def process_corpus(build_corpus_path):
  build_manifest = dataset_corpus.load_json_from_corpus(
      build_corpus_path, './build_manifest.json')
  if build_manifest is None:
    return None
  for target in build_manifest['targets']:
    if not target['success'] and target['build_log'] is not None:
      # We're assuming the spack builder here because that's mainly what this
      # script is being used for currently.
      # TODO(boomanaiden154): Make this more generic when #77 is fixed and the
      # corpora have been rebuilt.
      if build_corpus_path[-3:] == 'tar':
        build_log_path = f'{build_corpus_path}:./spack_build.log'
      else:
        build_log_path = target['build_log']
      return ('build_failure', target['name'], build_log_path)
    if target['build_log'] is None:
      return ('missing_logs', target['name'], None)
  return None


def main(_):
  ray.init()

  build_corpora = os.listdir(FLAGS.corpus_dir)
  corpus_futures = []

  for build_corpus in build_corpora:
    corpus_path = os.path.join(FLAGS.corpus_dir, build_corpus)
    corpus_futures.append(process_corpus.remote(corpus_path))

  build_failures = 0
  missing_logs = 0

  while len(corpus_futures) > 0:
    finished, corpus_futures = ray.wait(corpus_futures, timeout=5.0)
    finished_data = ray.get(finished)
    logging.info(
        f'Just finished {len(finished)}, {len(corpus_futures)} remaining.')
    for finished_corpus in finished_data:
      if finished_corpus is not None:
        if finished_corpus[0] == 'build_failure':
          build_failures += 1
          print(f'{finished_corpus[1]},failure,{finished_corpus[2]}')
        elif finished_corpus[0] == 'missing_logs':
          missing_logs += 1
          print(f'{finished_corpus[1]},no_logs,NULL')

  logging.warning(f'Found {build_failures} build failures.')
  logging.warning(f'{missing_logs} targets were missing logs.')


if __name__ == '__main__':
  app.run(main)
