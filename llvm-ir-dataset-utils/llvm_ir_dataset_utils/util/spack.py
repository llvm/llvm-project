"""Utilities related to spack."""

import subprocess
import os


def get_spack_arch_info(info_type):
  spack_arch_command_vector = ['spack', 'arch', f'--{info_type}']
  arch_process = subprocess.run(
      spack_arch_command_vector,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=True)
  return arch_process.stdout.decode('utf-8').rsplit()[0]


def get_compiler_version():
  compiler_command_vector = ['clang', '--version']
  compiler_version_process = subprocess.run(
      compiler_command_vector,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=True)
  version_line = compiler_version_process.stdout.decode('utf-8').split('\n')[0]
  version_line_parts = version_line.split(' ')
  for index, version_line_part in enumerate(version_line_parts):
    if version_line_part == 'version':
      return version_line_parts[index + 1]


def get_spack_compiler_config():
  compiler_config = (
      "compilers:\n"
      "- compiler:\n"
      f"    spec: clang@={get_compiler_version()}\n"
      "    paths:\n"
      "      cc: /tmp/llvm-ir-dataset-utils/utils/compiler_wrapper\n"
      "      cxx: /tmp/llvm-ir-dataset-utils/utils/compiler_wrapper++\n"
      "      f77: /usr/bin/gfortran\n"
      "      fc: /usr/bin/gfortran\n"
      "    flags:\n"
      "      cflags: -Xclang -fembed-bitcode=all\n"
      "      cxxflags: -Xclang -fembed-bitcode=all\n"
      f"    operating_system: {get_spack_arch_info('operating-system')}\n"
      "    target: x86_64\n"
      "    modules: []\n"
      "    environment: {}\n"
      "    extra_rpaths: []")
  return compiler_config


def get_spack_config(build_dir):
  spack_config = ("config:\n"
                  "  install_tree:\n"
                  f"    root: {build_dir}/spack-installs\n"
                  "    padded_length: 512\n"
                  "  build_stage:\n"
                  f"    - {build_dir}/build-stage\n"
                  f"  test_stage: {build_dir}/test-stage\n"
                  f"  source_cache: {build_dir}/source-cache\n"
                  f"  misc_cache: {build_dir}/misc-cache")
  return spack_config


def spack_setup_compiler(build_dir):
  compiler_config_path = os.path.join(build_dir, '.spack/compilers.yaml')
  with open(compiler_config_path, 'w') as compiler_config_file:
    compiler_config_file.writelines(get_spack_compiler_config())


def spack_setup_config(build_dir):
  spack_config_path = os.path.join(build_dir, '.spack/config.yaml')
  with open(spack_config_path, 'w') as spack_config_file:
    spack_config_file.writelines(get_spack_config(build_dir))
