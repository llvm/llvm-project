#!/usr/bin/env python

import argparse
import os
import platform
import settings
import shutil
import subprocess
import sys

from pathlib import Path, PurePath

def default_toolchain():
  system = platform.uname()[0]
  machine = platform.uname()[4]
  toolchain = Path(settings.THIS_DIR,
                           'cmake',
                           f'toolchain_{system.lower()}_{machine.lower()}.cmake')
  return toolchain if toolchain.is_file else None

def get_arguments():
  parser = argparse.ArgumentParser(
    description="Help configure and build classic-flang-llvm-project")

  buildopt = parser.add_argument_group('general build options')
  buildopt.add_argument('-t', '--target', metavar='ARCH', choices=['X86', 'AArch64', 'PowerPC'], default='X86',
                        help='Control which targets are enabled (%(choices)s) (default: %(default)s)')
  buildopt.add_argument('-p', '--install-prefix', metavar='PATH', nargs='?', default=None, const=False,
                        help='Install directory (default: do not install)')
  buildopt.add_argument('-j', '--jobs', metavar='N', type=int, default=os.cpu_count(),
                        help='number of parallel build jobs (default: %(default)s)')
  buildopt.add_argument('--toolchain', metavar='FILE', default=default_toolchain().as_posix(),
                        help='specify toolchain file (default: %(default)s)')
  buildopt.add_argument('-d', '--builddir', metavar='DIR', default='build',
                        help=f'specify build directory (default: {settings.LLVM_DIR}/%(default)s)')
  buildopt.add_argument('--clean', action='store_true', default=False,
                        help='clean build')
  buildopt.add_argument('-b', '--build-type', metavar='TYPE', default='Release',
                        help='set build type (default: %(default)s)')
  buildopt.add_argument('-x', '--cmake-param', metavar='OPT', action='append', default=[],
                        help='add custom argument to CMake')
  buildopt.add_argument('-e', '--llvm-enable-projects', metavar='OPT', action='append', default=['clang'] if sys.platform == 'win32' else ['clang', 'openmp'],
                        help='enable llvm projects to build (in quotation marks separated by semicolons e.g.: "clang;openmp")')
  buildopt.add_argument('-c', '--use-ccache', action="store_true", default=False,
                        help='Using ccache during the build (default: %(default)s)')
  buildopt.add_argument('--cc', metavar='OPT', default=None,
                        help='use specific C compiler')
  buildopt.add_argument('--cxx', metavar='OPT', default=None,
                        help='use specific C++ compiler')
  buildopt.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose build (default: %(default)s')
  arguments = parser.parse_args()
  return arguments

def generate_buildoptions(arguments):
  base_cmake_args = [
    f'-DCMAKE_BUILD_TYPE={arguments.build_type}',
    f'-DCMAKE_TOOLCHAIN_FILE={arguments.toolchain}'
  ]

  if sys.platform == 'win32' and platform.uname()[4].lower() == 'arm64':
    base_cmake_args.append('-GNMake Makefiles')
  else:
    generator = 'Ninja' if sys.platform == 'win32' else 'Unix Makefiles'
    base_cmake_args.append(f'-G{generator}')

  if arguments.install_prefix:
    install_root = Path(arguments.install_prefix)
    base_cmake_args.append(f'-DCMAKE_INSTALL_PREFIX={install_root.as_posix()}')

  if arguments.use_ccache:
    base_cmake_args.append('-DCMAKE_C_COMPILER_LAUNCHER=ccache')
    base_cmake_args.append('-DCMAKE_CXX_COMPILER_LAUNCHER=ccache')

  if arguments.cmake_param:
    base_cmake_args.extend(arguments.cmake_param)

  if arguments.cc:
    base_cmake_args.append(f'-DCMAKE_C_COMPILER={arguments.cc}')

  if arguments.cxx:
    base_cmake_args.append(f'-DCMAKE_CXX_COMPILER={arguments.cxx}')

  if arguments.verbose:
    base_cmake_args.append('-DCMAKE_VERBOSE_MAKEFILE=ON')

  return base_cmake_args

def normalize_builddir(project_srcdir, builddir, clean):
  build_path = ''
  if PurePath(builddir).is_absolute():
    build_path = Path(builddir, PurePath(project_srcdir).name)
  else:
    build_path = Path(project_srcdir, builddir)

  if clean and build_path.exists():
    shutil.rmtree(build_path)

  return build_path.as_posix()

def configure_llvm(arguments):
  build_path = normalize_builddir(
    settings.LLVM_DIR, arguments.builddir, arguments.clean)

  build_options = generate_buildoptions(arguments)
  additional_options = [
    f'-DLLVM_TARGETS_TO_BUILD={arguments.target}',
    '-DLLVM_ENABLE_CLASSIC_FLANG=ON',
  ]
  if(arguments.llvm_enable_projects):
    additional_options.append(f'-DLLVM_ENABLE_PROJECTS={";".join(x for x in arguments.llvm_enable_projects)}')
  build_options.extend(additional_options)
  cmake_cmd = ['cmake', '-B', build_path, '-S', Path.joinpath(settings.LLVM_DIR,'llvm')]
  cmake_cmd.extend(build_options)
  subprocess.run(cmake_cmd, check=True)

  return build_path

def build_project(build_path, arguments):
  build_cmd = ['cmake', '--build', build_path, '--config',
               arguments.build_type, '-j', str(arguments.jobs)]
  subprocess.run(build_cmd, check=True)

def install_project(build_path, arguments):
  install_cmd = ['cmake', '--build', build_path, '--config',
                 arguments.build_type, '--target', 'install']
  subprocess.run(install_cmd, check=True)

def print_success():
  print()
  print('=' * 30)
  print('Build succeeded!')
  print('=' * 30)
  print()

def print_header(title):
  print()
  print('*' * 30)
  print(f'{title}...')
  print('*' * 30)
  print()

def main():
  arguments = get_arguments()

  print_header('Building classic llvm')
  build_path = configure_llvm(arguments)
  build_project(build_path, arguments)
  if arguments.install_prefix is not None:
    install_project(build_path, arguments)

if __name__ == "__main__":
  main()
  print_success()
