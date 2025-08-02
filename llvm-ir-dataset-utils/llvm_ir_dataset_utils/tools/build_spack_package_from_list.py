"""A tool for building individual spack packages or an entire list from a list
of spack packages and their dependencies.
"""

import json

from absl import app
from absl import flags

import ray

from llvm_ir_dataset_utils.builders import builder

FLAGS = flags.FLAGS

flags.DEFINE_string('package_list', None, 'The list of spack packages and '
                    'their dependencies.')
flags.DEFINE_string('package_name', None, 'The name of an individual package '
                    'to build.')
flags.DEFINE_string('corpus_dir', None, 'The path to the corpus.')
flags.DEFINE_string(
    'source_dir', '/tmp/source', 'The source dir to pass along '
    'to the builder. This is not used by the spack builder.')
flags.DEFINE_string(
    'build_dir', None, 'The build dir to pass along to '
    'the builder. This is not used by the spack builder.')
flags.DEFINE_string(
    'buildcache_dir', None,
    'The directory of the spack buildcache to store built packages in.')
flags.DEFINE_integer('thread_count', 16,
                     'The number of threads to use per job.')
flags.DEFINE_bool(
    'archive_corpus', False,
    'Whether or not to put the output corpus for each package into an archive.')
flags.DEFINE_bool('cleanup', True,
                  'Whether or not to clean up the build directory')

flags.mark_flag_as_required('package_list')
flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('build_dir')
flags.mark_flag_as_required('buildcache_dir')


def get_package_future(package_dict, current_package_futures, package, threads):
  if package in current_package_futures:
    return current_package_futures[package]
  dependency_futures = []
  for dependency in package_dict[package]['deps']:
    if dependency in current_package_futures:
      dependency_futures.append(current_package_futures[dependency])
    else:
      dependency_futures.append(
          get_package_future(package_dict, current_package_futures, dependency,
                             threads))
  corpus_description = {
      'build_system': 'spack',
      'folder_name': f'{package_dict[package]["name"]}-{package}',
      'package_name': package_dict[package]['name'],
      'package_spec': package_dict[package]['spec'],
      'package_hash': package,
      'license': package_dict[package]['license'],
      'license_source': package_dict[package]['license_source'],
      'sources': []
  }
  extra_builder_arguments = {
      'dependency_futures': dependency_futures,
      'buildcache_dir': FLAGS.buildcache_dir
  }
  build_future = builder.get_build_future(
      corpus_description,
      FLAGS.source_dir,
      FLAGS.build_dir,
      FLAGS.corpus_dir,
      threads, {},
      extra_builder_arguments=extra_builder_arguments,
      cleanup=FLAGS.cleanup,
      archive_corpus=FLAGS.archive_corpus)
  current_package_futures[package] = build_future
  return build_future


def main(_):
  with open(FLAGS.package_list) as package_list_file:
    package_dict = json.load(package_list_file)

  ray.init()
  build_futures = []
  build_futures_dict = {}

  if FLAGS.package_name:
    package = None
    for package in package_dict:
      if package_dict[package]['name'] == FLAGS.package_name:
        break
    if package is None:
      raise ValueError('At least one package must be specified to be built.')
    build_futures.append(
        get_package_future(package_dict, build_futures_dict, package,
                           FLAGS.thread_count))
  else:
    for package in package_dict:
      build_future = get_package_future(package_dict, build_futures_dict,
                                        package, FLAGS.thread_count)
      build_futures.append(build_future)
      build_futures_dict[package] = build_future

  ray.get(build_futures)


if __name__ == '__main__':
  app.run(main)
