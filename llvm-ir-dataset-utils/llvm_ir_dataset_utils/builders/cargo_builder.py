"""Module for building and extracting bitcode from applications using cargo"""

import subprocess
import os
import json
import multiprocessing
import shutil
import pathlib
import logging

import ray

from mlgo.corpus import make_corpus_lib
from mlgo.corpus import combine_training_corpus_lib

BUILD_TIMEOUT = 900


def get_spec_from_id(id):
  sections = id.split('(')
  file_path = sections[1][5:-1]
  name_version = sections[0].split(' ')
  name = name_version[0]
  version = name_version[1]
  return f'{file_path}#{name}@{version}'


def get_packages_from_manifest(source_dir):
  command_vector = ["cargo", "metadata", "--no-deps"]
  if not os.path.exists(source_dir):
    return []
  try:
    # TODO(boomanaiden154): Dump the stderr of the metadata command to a log
    # somewhere
    out = subprocess.check_output(
        command_vector, cwd=source_dir, stderr=subprocess.PIPE)
    manifest = json.loads(out.decode("utf-8"))
    packages = {}
    for package in manifest["packages"]:
      targets = []
      for target in package["targets"]:
        targets.append({
            "name": target["name"],
            "kind": target["kind"][0],
            "spec": get_spec_from_id(package['id']),
            "package": package['name']
        })
      packages[package["name"]] = targets
    return packages
  except subprocess.SubprocessError:
    return []


def get_build_log_name(target):
  return './' + target['name'] + '.' + target['kind'] + '.build.log'


def build_all_targets(source_dir, build_dir, corpus_dir, threads,
                      extra_env_variables, cleanup):
  package_list = get_packages_from_manifest(source_dir)
  build_log = {'targets': []}
  package_futures = []
  for package in package_list:
    package_build_dir = build_dir + '-' + package
    package_futures.append(
        build_package_future(source_dir, package_build_dir, corpus_dir,
                             package_list[package], threads,
                             extra_env_variables, cleanup))
  package_build_logs = ray.get(package_futures)
  for package_build_log in package_build_logs:
    build_log['targets'].extend(package_build_log)
  combine_training_corpus_lib.combine_corpus(corpus_dir)
  return build_log


def build_package_future(source_dir, build_dir, corpus_dir, targets, threads,
                         extra_env_variables, cleanup):
  return build_package.options(num_cpus=threads).remote(source_dir, build_dir,
                                                        corpus_dir, targets,
                                                        threads,
                                                        extra_env_variables,
                                                        cleanup)


@ray.remote(num_cpus=multiprocessing.cpu_count())
def build_package(source_dir, build_dir, corpus_dir, targets, threads,
                  extra_env_variables, cleanup):
  build_log = []
  for target in targets:
    build_log.append(
        perform_build(source_dir, build_dir, corpus_dir, target, threads,
                      extra_env_variables))
  package_corpus_dir = os.path.join(corpus_dir, targets[0]["package"])
  # We should never be creating the parents of the folder as they should be
  # provided by builder.py and the folder should never exist before we create
  # it.
  pathlib.Path(package_corpus_dir).mkdir(exist_ok=False, parents=False)
  extract_ir(build_dir, package_corpus_dir)
  if cleanup:
    if os.path.exists(build_dir):
      try:
        shutil.rmtree(build_dir)
      except Exception:
        logging.warn(
            f'Failed to delete directory {build_dir}, probably deleted by another process.'
        )
  return build_log


def perform_build(source_dir, build_dir, corpus_dir, target, threads,
                  extra_env_variables):
  logging.info(
      f"Building target {target['name']} of type {target['kind']} from package {target['package']}"
  )
  build_env = os.environ.copy()
  build_env["CARGO_TARGET_DIR"] = build_dir
  build_env.update(extra_env_variables)
  build_command_vector = [
      "cargo", "rustc", "-p", f"{target['spec']}", "-j",
      str(threads)
  ]
  if target['kind'] == "lib":
    build_command_vector.append("--lib")
  elif target['kind'] == "test":
    build_command_vector.extend(["--test", target['name']])
  elif target['kind'] == "bench":
    build_command_vector.extend(["--bench", target['name']])
  elif target['kind'] == "bin":
    build_command_vector.extend(["--bin", target['name']])
  elif target['kind'] == "example":
    build_command_vector.extend(["--example", target['name']])
  else:
    logging.warn(
        f'{target["name"]} has unrecognized target type {target["kind"]} in package {target["package"]}'
    )
    return {
        'success': False,
        'build_log': None,
        'name': target['name'] + '.' + target['kind']
    }
  build_command_vector.extend(
      ["--", '--emit=llvm-bc', '-C', 'no-prepopulate-passes'])
  try:
    build_log_path = os.path.join(corpus_dir, get_build_log_name(target))
    with open(build_log_path, 'w') as build_log_file:
      subprocess.run(
          build_command_vector,
          cwd=source_dir,
          env=build_env,
          check=True,
          stdout=build_log_file,
          stderr=build_log_file,
          timeout=BUILD_TIMEOUT)
  except subprocess.SubprocessError:
    logging.warn(
        f"Failed to build target {target['name']} of type {target['kind']} from package {target['package']}"
    )
    build_success = False
  else:
    logging.info(
        f"Finished building target {target['name']} of type {target['kind']} from package {target['package']}"
    )
    build_success = True
  return {
      'success': build_success,
      'build_log': get_build_log_name(target),
      'name': target['name'] + '.' + target['kind']
  }


def extract_ir(build_dir, corpus_dir):
  # TODO(boomanaiden154): Look into getting a build manifest from cargo.
  relative_paths = make_corpus_lib.load_bitcode_from_directory(build_dir)
  make_corpus_lib.copy_bitcode(relative_paths, build_dir, corpus_dir)
  make_corpus_lib.write_corpus_manifest(relative_paths, corpus_dir, '')
