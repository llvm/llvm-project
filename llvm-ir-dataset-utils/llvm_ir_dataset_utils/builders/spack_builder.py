"""Module for building and extracting bitcode from applications using spack"""

import subprocess
import os
import tempfile
import logging
import pathlib
import shutil
import re
import getpass

import ray

from mlgo.corpus import extract_ir_lib

from llvm_ir_dataset_utils.util import file
from llvm_ir_dataset_utils.util import spack as spack_utils
from llvm_ir_dataset_utils.util import extract_source_lib

SPACK_THREAD_OVERSUBSCRIPTION_FACTOR = 1

SPACK_GARBAGE_COLLECTION_TIMEOUT = 300

BUILD_LOG_NAME = './spack_build.log'


def get_spec_command_vector_section(spec):
  filtered_spec = re.sub(r'license=".*?" ', '', spec)
  # Strip the patches list from a package that we're pushing to a build cache.
  # There is at least one case where Spack fails to match the package for pushing
  # to the buildcache after installation due to the patches string.
  # TODO(boomanaiden154): Investigate why this is and remove it once this gets
  # fixed.
  filtered_spec2 = re.sub(r'patches=.*? ', '', filtered_spec)
  return filtered_spec2.split(' ')


def generate_build_command(package_to_build, threads, build_dir):
  command_vector = [
      'spack', '--insecure', '-c', f'config:build_stage:{build_dir}', 'install',
      '--keep-stage', '--overwrite', '-y', '--use-buildcache',
      'package:never,dependencies:only', '-j',
      f'{SPACK_THREAD_OVERSUBSCRIPTION_FACTOR * threads}',
      '--no-check-signature', '--deprecated'
  ]
  command_vector.extend(get_spec_command_vector_section(package_to_build))
  return command_vector


def perform_build(package_name, assembled_build_command, corpus_dir, build_dir):
  logging.info(f"Spack building package {package_name}")
  environment = os.environ.copy()
  # Set $HOME to the build directory so that spack doesn't run into weird
  # errors with multiple machines trying to write to a common home directory.
  environment['HOME'] = build_dir
  build_log_path = os.path.join(corpus_dir, BUILD_LOG_NAME)
  try:
    with open(build_log_path, 'w') as build_log_file:
      subprocess.run(
          assembled_build_command,
          stdout=build_log_file,
          stderr=build_log_file,
          check=True,
          env=environment)
  except subprocess.SubprocessError:
    logging.warn(f"Failed to build spack package {package_name}")
    return False
  logging.info(f"Finished build spack package {package_name}")
  return True


def get_spack_stage_directory(package_hash, build_dir):
  spack_build_directory = os.path.join(build_dir, getpass.getuser())
  if not os.path.exists(spack_build_directory):
    return None
  spack_stages = os.listdir(spack_build_directory)
  spack_stages.append('')
  for spack_stage_dir in spack_stages:
    if package_hash in spack_stage_dir:
      break
  # spack_stage_dir now contains the name of the directory,  or None if we
  # failed to find a stage directory (i.e., due to build failure)
  if spack_stage_dir == '':
    logging.warning(f'Failed to get stage dir for {package_hash}. This might '
                    'have been caused by your spack installation and the '
                    'package_list.json becoming out of sync.')
    return None
  return os.path.join(spack_build_directory, spack_stage_dir)


def extract_ir(package_hash, corpus_dir, build_dir, threads):
  build_directory = get_spack_stage_directory(package_hash, build_dir)
  if build_directory is not None:
    current_verbosity = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)
    objects = extract_ir_lib.load_from_directory(build_directory, corpus_dir)
    relative_output_paths = extract_ir_lib.run_extraction(
        objects, threads, "llvm-objcopy", None, None, ".llvmcmd", ".llvmbc")
    extract_ir_lib.write_corpus_manifest(None, relative_output_paths,
                                         corpus_dir)
    logging.getLogger().setLevel(current_verbosity)
    extract_source_lib.copy_source(build_directory, corpus_dir)


def push_to_buildcache(package_spec, buildcache_dir, corpus_dir, build_dir):
  command_vector = [
      'spack', 'buildcache', 'push', '--unsigned', '--only', 'package',
      buildcache_dir
  ]
  command_vector.extend(get_spec_command_vector_section(package_spec))
  buildcache_push_log_path = os.path.join(corpus_dir, 'buildcache_push.log')
  environment = os.environ.copy()
  environment['HOME'] = build_dir
  with open(buildcache_push_log_path, 'w') as buildcache_push_log_file:
    subprocess.run(
        command_vector,
        check=True,
        env=environment,
        stdout=buildcache_push_log_file,
        stderr=buildcache_push_log_file)


def cleanup(package_name, package_spec, corpus_dir, build_dir, uninstall=True):
  environment = os.environ.copy()
  environment['HOME'] = build_dir
  if uninstall:
    uninstall_command_vector = ['spack', 'uninstall', '-y']
    uninstall_command_vector.extend(
        get_spec_command_vector_section(package_spec))
    uninstall_log_path = os.path.join(corpus_dir, 'uninstall.log')
    with open(uninstall_log_path, 'w') as uninstall_log_file:
      subprocess.run(
          uninstall_command_vector,
          check=True,
          env=environment,
          stdout=uninstall_log_file,
          stderr=uninstall_log_file)
  # Garbage collect dependencies
  try:
    gc_command_vector = ['spack', 'gc', '-y']
    gc_log_path = os.path.join(corpus_dir, 'gc.log')
    with open(gc_log_path, 'w') as gc_log_file:
      subprocess.run(
          gc_command_vector,
          check=True,
          env=environment,
          stdout=gc_log_file,
          stderr=gc_log_file,
          timeout=SPACK_GARBAGE_COLLECTION_TIMEOUT)
  except subprocess.SubprocessError:
    logging.warning(
        f'Failed to garbage collect while cleaning up package {package_name}.')


def construct_build_log(build_success, package_name):
  return {
      'targets': [{
          'name': package_name,
          'build_log': BUILD_LOG_NAME,
          'success': build_success
      }]
  }


def spack_add_mirror(build_dir, buildcache_dir):
  environment = os.environ.copy()
  environment['HOME'] = build_dir
  command_vector = ['spack', 'mirror', 'add', 'buildcache', buildcache_dir]
  subprocess.run(
      command_vector,
      check=True,
      env=environment,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL)


def spack_setup_bootstrap_root(build_dir):
  # TODO(boomanaiden154): Pull out the hardcoded /tmp/spack-boostrap path and
  # make it a configurable somewhere.
  bootstrap_dir = os.path.join(build_dir, 'spack-bootstrap')
  shutil.copytree('/tmp/spack-bootstrap', bootstrap_dir)
  command_vector = ['spack', 'bootstrap', 'root', bootstrap_dir]
  environment = os.environ.copy()
  environment['HOME'] = build_dir
  subprocess.run(
      command_vector,
      env=environment,
      check=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL)


def build_package(dependency_futures,
                  package_name,
                  package_spec,
                  package_hash,
                  corpus_dir,
                  threads,
                  buildcache_dir,
                  build_dir,
                  cleanup_build=False):
  dependency_futures = ray.get(dependency_futures)
  for dependency_future in dependency_futures:
    if not dependency_future['targets'][0]['success']:
      logging.warning(
          f'Dependency {dependency_future["targets"][0]["name"]} failed to build'
          f'for package{package_name}, not building.')
      if cleanup_build:
        cleanup(
            package_name, package_spec, corpus_dir, build_dir, uninstall=False)
      return construct_build_log(False, package_name)
  spack_add_mirror(build_dir, buildcache_dir)
  spack_utils.spack_setup_compiler(build_dir)
  spack_utils.spack_setup_config(build_dir)
  spack_setup_bootstrap_root(build_dir)
  build_command = generate_build_command(package_spec, threads, build_dir)
  build_result = perform_build(package_name, build_command, corpus_dir,
                               build_dir)
  if build_result:
    extract_ir(package_hash, corpus_dir, build_dir, threads)
    push_to_buildcache(package_spec, buildcache_dir, corpus_dir, build_dir)
    logging.warning(f'Finished building {package_name}')
  if cleanup_build:
    if build_result:
      cleanup(package_name, package_spec, corpus_dir, build_dir, package_hash)
    else:
      cleanup(
          package_name, package_spec, corpus_dir, build_dir, uninstall=False)
  return construct_build_log(build_result, package_name)
