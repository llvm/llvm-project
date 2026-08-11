"""Tool for getting all spack packages that are usable for producing LLVM
bitcode.

Note: This must be run with `spack-python` or `spack python` rather than your
default python interpreter.
"""

import json
import multiprocessing
import tempfile
import os
import subprocess
import logging
import sys
import re

from absl import app
from absl import flags

import ray

import spack.repo
import spack.environment
import spack.spec
import spack.config

from llvm_ir_dataset_utils.util import spack as spack_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('package_list', 'package_list.json',
                    'The path to write the package list to.')
flags.DEFINE_string(
    'error_log', None,
    'The path to write the output of failed concretization commands to.')
flags.DEFINE_integer('max_projects', sys.maxsize,
                     'The max number of projects to process.')


def add_concrete_package_and_all_deps(concretized_packages, spec):
  spec_string = str(spec)
  license_string = re.findall('license=".*?"', spec_string)[0][9:-1]
  license_source = None
  if license_string != 'NOASSERTION':
    license_source = 'spack'
  concretized_packages[spec.dag_hash()] = {
      'spec': spec_string,
      'deps': [dep_spec.dag_hash() for dep_spec in spec.dependencies()],
      'name': str(spec.package.fullname.split('.')[1]),
      'license': license_string,
      'license_source': license_source
  }
  for dep_spec in spec.dependencies():
    if dep_spec.dag_hash() not in concretized_packages:
      add_concrete_package_and_all_deps(concretized_packages, dep_spec)


@ray.remote(num_cpus=1)
def concretize_environment(package_name):
  concretized_packages = {}
  with tempfile.TemporaryDirectory() as tempdir:
    env = spack.environment.create_in_dir(tempdir)
    env.add(spack.spec.Spec(package_name))
    env.unify = False
    env.write()

    os.mkdir(os.path.join(tempdir, '.spack'))
    command_env = os.environ.copy()
    command_env['HOME'] = tempdir
    spack_utils.spack_setup_compiler(tempdir)

    concretize_command_vector = ['spack', '-e', './', 'concretize']

    command_output = subprocess.run(
        concretize_command_vector,
        cwd=tempdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=command_env,
        universal_newlines=True)

    if command_output.returncode == 0:
      env = spack.environment.Environment(tempdir)

      concretized_specs = env.all_specs()
      for concretized_spec in concretized_specs:
        add_concrete_package_and_all_deps(concretized_packages,
                                          concretized_spec)
      return (command_output.stdout, concretized_packages, package_name)
    else:
      return (command_output.stdout, None, package_name)


def get_concretization_future(package_name):
  return concretize_environment.remote(package_name)


def main(_):
  ray.init()
  logging.info('Getting packages.')
  packages = spack.repo.all_package_names(include_virtuals=True)

  full_package_list = []

  for package in packages:
    pkg_class = spack.repo.PATH.get_pkg_class(package)
    # TODO(boomanaiden154): Look into other build systems that are likely to be
    # composed of c/c++ projects.
    pkg = pkg_class(spack.spec.Spec(package))
    if (pkg.build_system_class == 'CMakePackage' or
        pkg.build_system_class == 'MakefilePackage' or
        pkg.build_system_class == 'AutotoolsPackage' or
        pkg.build_system_class == 'MesonPackage'):
      full_package_list.append(pkg.name)

    if len(full_package_list) >= FLAGS.max_projects:
      break

  logging.info('Concretizing packages')
  concretization_futures = []
  for package in full_package_list:
    concretization_futures.append(get_concretization_future(package))

  concretized_packages = {}

  error_log_file = None

  if FLAGS.error_log is not None:
    error_log_file = open(FLAGS.error_log, 'w')

  while len(concretization_futures) > 0:
    finished, concretization_futures = ray.wait(
        concretization_futures, timeout=5.0)
    finished_data = ray.get(finished)
    for data in finished_data:
      if data[1] is None:
        if error_log_file is not None:
          error_log_file.write(
              f'Encountered the following errors while concretizing {data[2]}:\n'
          )
          error_log_file.write(data[0])
      else:
        concretized_packages.update(data[1])
    logging.info(
        f'Just finished {len(finished_data)}, {len(concretization_futures)} remaining'
    )

  if error_log_file is not None:
    error_log_file.close()

  with open(FLAGS.package_list, 'w') as package_list_file:
    json.dump(concretized_packages, package_list_file, indent=2)


if __name__ == '__main__':
  app.run(main)
