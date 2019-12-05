# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import platform
import shlex
import shutil

import lit.formats

# name: The name of this test suite.
config.name = 'lldb-api'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_source_root = os.path.join(config.lldb_src_root, 'packages',
                                       'Python', 'lldbsuite', 'test')
config.test_exec_root = config.test_source_root

if 'Address' in config.llvm_use_sanitizer:
  config.environment['ASAN_OPTIONS'] = 'detect_stack_use_after_return=1'
  # macOS flags needed for LLDB built with address sanitizer.
  if 'Darwin' in config.host_os and 'x86' in config.host_triple:
    import subprocess
    resource_dir = subprocess.check_output(
        [config.cmake_cxx_compiler,
         '-print-resource-dir']).decode('utf-8').strip()
    runtime = os.path.join(resource_dir, 'lib', 'darwin',
                           'libclang_rt.asan_osx_dynamic.dylib')
    config.environment['DYLD_INSERT_LIBRARIES'] = runtime

def find_shlibpath_var():
  if platform.system() in ['Linux', 'FreeBSD', 'NetBSD', 'SunOS']:
    yield 'LD_LIBRARY_PATH'
  elif platform.system() == 'Darwin':
    yield 'DYLD_LIBRARY_PATH'
  elif platform.system() == 'Windows':
    yield 'PATH'

# Shared library build of LLVM may require LD_LIBRARY_PATH or equivalent.
if config.shared_libs:
  for shlibpath_var in find_shlibpath_var():
    # In stand-alone build llvm_shlib_dir specifies LLDB's lib directory while
    # llvm_libs_dir specifies LLVM's lib directory.
    shlibpath = os.path.pathsep.join(
        (config.llvm_shlib_dir, config.llvm_libs_dir,
         config.environment.get(shlibpath_var, '')))
    config.environment[shlibpath_var] = shlibpath
  else:
    lit_config.warning("unable to inject shared library path on '{}'".format(
        platform.system()))

# Propagate LLDB_CAPTURE_REPRODUCER
if 'LLDB_CAPTURE_REPRODUCER' in os.environ:
  config.environment['LLDB_CAPTURE_REPRODUCER'] = os.environ[
      'LLDB_CAPTURE_REPRODUCER']

# Clean the module caches in the test build directory. This is necessary in an
# incremental build whenever clang changes underneath, so doing it once per
# lit.py invocation is close enough.
for cachedir in [config.clang_module_cache, config.lldb_module_cache]:
  if os.path.isdir(cachedir):
    print("Deleting module cache at %s."%cachedir)
    shutil.rmtree(cachedir)

# Set a default per-test timeout of 10 minutes. Setting a timeout per test
# requires that killProcessAndChildren() is supported on the platform and
# lit complains if the value is set but it is not supported.
supported, errormsg = lit_config.maxIndividualTestTimeIsSupported
if supported:
  lit_config.maxIndividualTestTime = 600
else:
  lit_config.warning("Could not set a default per-test timeout. " + errormsg)

# Build dotest command.
dotest_cmd = [config.dotest_path]
dotest_cmd.extend(config.dotest_args_str.split(';'))

# We don't want to force users passing arguments to lit to use `;` as a
# separator. We use Python's simple lexical analyzer to turn the args into a
# list.
if config.dotest_lit_args_str:
  dotest_cmd.extend(shlex.split(config.dotest_lit_args_str))

# Library path may be needed to locate just-built clang.
if config.llvm_libs_dir:
  dotest_cmd += ['--env', 'LLVM_LIBS_DIR=' + config.llvm_libs_dir]

if config.lldb_build_directory:
  dotest_cmd += ['--build-dir', config.lldb_build_directory]

if config.lldb_module_cache:
  dotest_cmd += ['--lldb-module-cache-dir', config.lldb_module_cache]

if config.clang_module_cache:
  dotest_cmd += ['--clang-module-cache-dir', config.clang_module_cache]

# Load LLDB test format.
sys.path.append(os.path.join(config.lldb_src_root, "test", "API"))
import lldbtest

# testFormat: The test format to use to interpret tests.
config.test_format = lldbtest.LLDBTest(dotest_cmd)
