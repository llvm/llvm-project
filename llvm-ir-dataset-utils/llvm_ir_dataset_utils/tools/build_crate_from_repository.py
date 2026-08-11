"""Tool to build a crate given just a repository."""

import json
import logging

from absl import app
from absl import flags
import ray

from llvm_ir_dataset_utils.builders import builder

FLAGS = flags.FLAGS

flags.DEFINE_string('repository', None, 'The repository url to clone from.')
flags.DEFINE_string('repository_list', None,
                    'Path to a file containing a list of repositories.')
flags.DEFINE_string('source_dir', None,
                    'The directory to download source code into.')
flags.DEFINE_string('build_dir', None,
                    'The base directory to and perform builds in.')
flags.DEFINE_string('corpus_dir', None, 'The directory to place the corpus in.')
flags.DEFINE_integer('thread_count', 8,
                     'The number of threads to use per crate build.')
flags.DEFINE_string('cargo_home', '/cargo', 'The default cargo directory.')
flags.DEFINE_string('rustup_home', '/rustup',
                    'The default rustup home directory.')
flags.DEFINE_bool(
    'archive_corpus', False,
    'Whether or not to put the output corpus for each package into an archive.')

flags.mark_flag_as_required('source_dir')
flags.mark_flag_as_required('build_dir')
flags.mark_flag_as_required('corpus_dir')


@flags.multi_flags_validator(
    ['repository', 'repository_list'],
    message=(
        'Expected one and only one of --repository and --repository_list to be'
        'defined.'),
)
def _validate_input_columns(flags_dict):
  both_defined = flags_dict['repository'] is not None and flags_dict[
      'repository_list'] is not None
  neither_defined = flags_dict['repository'] is None and flags_dict[
      'repository_list'] is None
  return not both_defined and not neither_defined


def main(_):
  ray.init()
  crates_list = []
  if FLAGS.repository is not None:
    crates_list.append(FLAGS.repository)
  elif FLAGS.repository_list is not None:
    with open(FLAGS.repository_list) as repository_list_file:
      crates_list = json.load(repository_list_file)

  build_futures = []
  for index, crate_to_build in enumerate(crates_list):
    sources = []
    if crate_to_build['repository'] is not None:
      sources.append({
          'type': 'git',
          'repo_url': crate_to_build['repository'],
          'commit_sha': ''
      })
    if crate_to_build['tar_archive'] is not None:
      sources.append({
          'type': 'tar',
          'archive_url': crate_to_build['tar_archive']
      })
    corpus_description = {
        'sources': sources,
        'folder_name': f'build-{index}',
        'build_system': 'cargo',
        'license': crate_to_build['license'],
        'license_source': crate_to_build['license_source']
    }

    additional_build_env_variables = {
        'RUSTUP_HOME': FLAGS.rustup_home,
        'CARGO_HOME': FLAGS.cargo_home
    }

    build_futures.append(
        builder.get_build_future(
            corpus_description,
            FLAGS.source_dir,
            FLAGS.build_dir,
            FLAGS.corpus_dir,
            FLAGS.thread_count,
            additional_build_env_variables,
            cleanup=True,
            archive_corpus=FLAGS.archive_corpus))

  all_finished = []
  while len(build_futures) > 0:
    finished, build_futures = ray.wait(build_futures, timeout=5.0)
    finished_data = ray.get(finished)
    all_finished.extend(finished_data)
    logging.info(
        f'Just finished {len(finished_data)}, {len(all_finished)} done, {len(build_futures)} remaining'
    )


if __name__ == '__main__':
  app.run(main)
