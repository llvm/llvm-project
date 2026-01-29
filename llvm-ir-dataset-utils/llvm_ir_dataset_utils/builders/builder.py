"""Module that parses application description, downloads source code, and invokes the correct builder"""

import glob
import hashlib
import json
import logging
import multiprocessing
import os
import pathlib
import shutil

import ray

from llvm_ir_dataset_utils.builders import (
    autoconf_builder,
    cargo_builder,
    cmake_builder,
    julia_builder,
    manual_builder,
    spack_builder,
    swift_builder,
)
from llvm_ir_dataset_utils.sources import source
from llvm_ir_dataset_utils.util import file, licenses


def get_corpus_size(corpus_dir):
  total_size = 0
  for bitcode_file in glob.glob(
      os.path.join(corpus_dir, "**/*.bc"), recursive=True):
    total_size += os.path.getsize(bitcode_file)
  return total_size


def get_corpus_source_size(corpus_dir):
  total_source_size = 0
  for source_file in glob.glob(
      os.path.join(corpus_dir, "**/*.source"), recursive=True):
    total_source_size += os.path.getsize(source_file)
  total_preprocessed_source_size = 0
  for preprocessed_source_file in glob.glob(
      os.path.join(corpus_dir, "**/*.preprocessed_source"), recursive=True):
    total_preprocessed_source_size += os.path.getsize(preprocessed_source_file)
  return (total_source_size, total_preprocessed_source_size)


def get_build_future(
    corpus_description,
    source_base_dir,
    build_base_dir,
    corpus_dir,
    threads,
    extra_env_variables,
    extra_builder_arguments={},
    cleanup=False,
    archive_corpus=False,
):
  return parse_and_build_from_description.options(num_cpus=threads).remote(
      corpus_description,
      source_base_dir,
      build_base_dir,
      corpus_dir,
      threads,
      extra_env_variables,
      extra_builder_arguments=extra_builder_arguments,
      cleanup=cleanup,
      archive_corpus=archive_corpus,
  )


def get_license_information(source_dir, corpus_dir):
  license_files = licenses.get_all_license_files(source_dir)
  license_file_list = []
  for license_description in license_files:
    # copy each license over
    license_file = license_description["file"]
    with open(os.path.join(source_dir, license_file),
              "rb") as license_file_handle:
      license_data = license_file_handle.read()
      license_hash = hashlib.sha256(license_data).hexdigest()
      new_license_path = f"./license-{license_hash}.txt"
      new_license_dict = license_description
      new_license_dict["file"] = new_license_path
      license_file_list.append(new_license_dict)
    with open(os.path.join(corpus_dir, new_license_path),
              "wb") as new_license_file_handle:
      new_license_file_handle.write(license_data)
  return license_file_list


@ray.remote(num_cpus=multiprocessing.cpu_count())
def parse_and_build_from_description(
    corpus_description,
    source_base_dir,
    build_base_dir,
    corpus_base_dir,
    threads,
    extra_env_variables,
    extra_builder_arguments={},
    cleanup=False,
    archive_corpus=False,
):
  # Construct relevant paths for the build
  corpus_dir = os.path.join(corpus_base_dir, corpus_description["folder_name"])
  if corpus_description["build_system"] == "manual":
    build_dir = os.path.join(build_base_dir, corpus_description["folder_name"])
  else:
    build_dir = os.path.join(build_base_dir,
                             corpus_description["folder_name"] + "-build")
  source_dir = os.path.join(source_base_dir, corpus_description["folder_name"])

  # Handle the case where we are archiving corpora and we already have some
  # packages that have finished building.
  if archive_corpus and os.path.exists(f"{corpus_dir}.tar"):
    # We already have an archived corpus for this package, so we can exit early
    # without doing the build.
    logging.warning(
        f"Found already built version of package at {corpus_dir}, skipping")
    return {}
  else:
    if os.path.exists(corpus_dir):
      shutil.rmtree(corpus_dir, ignore_errors=True)
    if os.path.exists(build_dir):
      shutil.rmtree(build_dir)
    if os.path.exists(source_dir):
      shutil.rmtree(source_dir)

  pathlib.Path(corpus_dir).mkdir(exist_ok=True, parents=True)
  pathlib.Path(source_base_dir).mkdir(exist_ok=True)
  pathlib.Path(build_base_dir).mkdir(exist_ok=True)
  to_download_dir = (
      build_base_dir
      if corpus_description["build_system"] == "manual" else source_base_dir)
  source_logs = source.download_source(
      corpus_description["sources"],
      to_download_dir,
      corpus_dir,
      corpus_description["folder_name"],
  )

  if not os.path.exists(build_dir):
    os.makedirs(build_dir)
  build_log = {}
  if corpus_description["build_system"] == "cmake":
    configure_command_vector = cmake_builder.generate_configure_command(
        os.path.join(source_dir, corpus_description["cmake_root"]),
        corpus_description["cmake_flags"],
    )
    build_command_vector = cmake_builder.generate_build_command([], threads)
    build_log = cmake_builder.perform_build(configure_command_vector,
                                            build_command_vector, build_dir,
                                            corpus_dir)
    cmake_builder.extract_ir(build_dir, corpus_dir, threads)
  elif corpus_description["build_system"] == "manual":
    if "environment_variables" in corpus_description:
      environment_variables = corpus_description["environment_variables"]
    else:
      environment_variables = {}
    build_log = manual_builder.perform_build(
        corpus_description["commands"],
        build_dir,
        threads,
        corpus_dir,
        environment_variables,
    )
    manual_builder.extract_ir(build_dir, corpus_dir, threads)
    if "raw_bc_corpus" in corpus_description:
      bc_corpus_dir = f'{corpus_dir}-{corpus_description["raw_bc_corpus"]}'
      os.makedirs(bc_corpus_dir)
      manual_builder.extract_raw_ir(build_dir, bc_corpus_dir, threads)
  elif corpus_description["build_system"] == "autoconf":
    configure_command_vector = autoconf_builder.generate_configure_command(
        source_dir, corpus_description["autoconf_flags"])
    build_command_vector = autoconf_builder.generate_build_command(threads)
    build_log = autoconf_builder.perform_build(configure_command_vector,
                                               build_command_vector, build_dir,
                                               corpus_dir)
    autoconf_builder.extract_ir(build_dir, corpus_dir, threads)
  elif corpus_description["build_system"] == "cargo":
    build_log = cargo_builder.build_all_targets(source_dir, build_dir,
                                                corpus_dir, threads,
                                                extra_env_variables, cleanup)
    if len(build_log["targets"]) == 0 and source_logs[-1]["type"] == "git":
      logging.warn("Cargo builder detected no targets from git repository, "
                   "retrying with tar archive.")
      shutil.rmtree(source_dir)
      # The git repository is always guaranteed to be the first source as long
      # as parse_crates_database.py was the source
      corpus_description["sources"].pop(0)
      build_future = get_build_future(
          corpus_description,
          source_base_dir,
          build_base_dir,
          corpus_base_dir,
          threads,
          extra_env_variables,
          cleanup,
      )
      ray.get(build_future)
      return {}
  elif corpus_description["build_system"] == "spack":
    if "dependency_futures" in extra_builder_arguments:
      dependency_futures = extra_builder_arguments["dependency_futures"]
    else:
      dependency_futures = []
    build_log = spack_builder.build_package(
        dependency_futures,
        corpus_description["package_name"],
        corpus_description["package_spec"],
        corpus_description["package_hash"],
        corpus_dir,
        threads,
        extra_builder_arguments["buildcache_dir"],
        build_dir,
        cleanup,
    )
  elif corpus_description["build_system"] == "julia":
    build_log = julia_builder.perform_build(corpus_description["package_name"],
                                            build_dir, corpus_dir, threads)
  elif corpus_description["build_system"] == "swift":
    build_log = swift_builder.perform_build(
        source_dir,
        build_dir,
        corpus_dir,
        threads,
        corpus_description["package_name"],
    )
  else:
    raise ValueError(
        f"Build system {corpus_description['build_system']} is not supported")

  # Collect license files from the build
  source_license_dir = source_dir
  if corpus_description["build_system"] == "spack":
    # Spack doesn't use the source directory, so we should instead pull
    # information from the build directory.
    spack_stage_dir = spack_builder.get_spack_stage_directory(
        corpus_description["package_hash"], build_dir)
    if spack_stage_dir is None:
      source_license_dir = None
    else:
      source_license_dir = os.path.join(spack_stage_dir, "spack-src")
  elif corpus_description["build_system"] == "manual":
    # The manual builder clones everything into the build directory, so
    # just use that.
    source_license_dir = build_dir
  if source_license_dir is not None:
    build_log["license_files"] = get_license_information(
        source_license_dir, corpus_dir)
  else:
    build_log["license_files"] = []

  if cleanup:
    file.delete_directory(build_dir, corpus_dir)
    file.delete_directory(source_dir, corpus_dir)
  build_log["sources"] = source_logs
  build_log["size"] = get_corpus_size(corpus_dir)

  source_size, preprocessed_source_size = get_corpus_source_size(corpus_dir)
  build_log["source_size"] = source_size
  build_log["preprocessed_source_size"] = preprocessed_source_size

  if "license" in corpus_description:
    build_log["license"] = corpus_description["license"]
  else:
    build_log["license"] = None

  if "license_source" in corpus_description:
    build_log["license_source"] = corpus_description["license_source"]
  else:
    build_log["license_source"] = None

  with open(os.path.join(corpus_dir, "build_manifest.json"),
            "w") as build_manifest:
    json.dump(build_log, build_manifest, indent=2)
  if archive_corpus:
    # Use corpus_dir for the file path as make_archive automatically adds the
    # .tar extension to the path
    shutil.make_archive(corpus_dir, "tar", corpus_dir)
    shutil.rmtree(corpus_dir, ignore_errors=True)
  return build_log
