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
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root


def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.isdir(path):
        raise OSError(errno.ENOTDIR, "%s is not a directory"%path)


def find_sanitizer_runtime(name):
  import subprocess
  resource_dir = subprocess.check_output(
      [config.cmake_cxx_compiler,
       '-print-resource-dir']).decode('utf-8').strip()
  return os.path.join(resource_dir, 'lib', 'darwin', name)


def find_shlibpath_var():
  if platform.system() in ['Linux', 'FreeBSD', 'NetBSD', 'SunOS']:
    yield 'LD_LIBRARY_PATH'
  elif platform.system() == 'Darwin':
    yield 'DYLD_LIBRARY_PATH'
  elif platform.system() == 'Windows':
    yield 'PATH'


# On macOS, we can't do the DYLD_INSERT_LIBRARIES trick with a shim python
# binary as the ASan interceptors get loaded too late. Also, when SIP is
# enabled, we can't inject libraries into system binaries at all, so we need a
# copy of the "real" python to work with.
def find_python_interpreter():
  # Avoid doing any work if we already copied the binary.
  copied_python = os.path.join(config.lldb_build_directory, 'copied-python')
  if os.path.isfile(copied_python):
    return copied_python

  # Find the "real" python binary.
  import shutil, subprocess
  real_python = subprocess.check_output([
      config.python_executable,
      os.path.join(os.path.dirname(os.path.realpath(__file__)),
                   'get_darwin_real_python.py')
  ]).decode('utf-8').strip()

  shutil.copy(real_python, copied_python)

  # Now make sure the copied Python works. The Python in Xcode has a relative
  # RPATH and cannot be copied.
  try:
    # We don't care about the output, just make sure it runs.
    subprocess.check_output([copied_python, '-V'], stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError:
    # The copied Python didn't work. Assume we're dealing with the Python
    # interpreter in Xcode. Given that this is not a system binary SIP
    # won't prevent us form injecting the interceptors so we get away with
    # not copying the executable.
    os.remove(copied_python)
    return real_python

  # The copied Python works.
  return copied_python


if 'Address' in config.llvm_use_sanitizer:
  config.environment['ASAN_OPTIONS'] = 'detect_stack_use_after_return=1'
  if 'Darwin' in config.host_os and 'x86' in config.host_triple:
    config.environment['DYLD_INSERT_LIBRARIES'] = find_sanitizer_runtime(
        'libclang_rt.asan_osx_dynamic.dylib')

if 'Thread' in config.llvm_use_sanitizer:
  if 'Darwin' in config.host_os and 'x86' in config.host_triple:
    config.environment['DYLD_INSERT_LIBRARIES'] = find_sanitizer_runtime(
        'libclang_rt.tsan_osx_dynamic.dylib')

if 'DYLD_INSERT_LIBRARIES' in config.environment and platform.system() == 'Darwin':
  config.python_executable = find_python_interpreter()

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

# Support running the test suite under the lldb-repro wrapper. This makes it
# possible to capture a test suite run and then rerun all the test from the
# just captured reproducer.
lldb_repro_mode = lit_config.params.get('lldb-run-with-repro', None)
if lldb_repro_mode:
  lit_config.note("Running API tests in {} mode.".format(lldb_repro_mode))
  mkdir_p(config.lldb_reproducer_directory)
  if lldb_repro_mode == 'capture':
    config.available_features.add('lldb-repro-capture')
  elif lldb_repro_mode == 'replay':
    config.available_features.add('lldb-repro-replay')

# Clean the module caches in the test build directory. This is necessary in an
# incremental build whenever clang changes underneath, so doing it once per
# lit.py invocation is close enough.
for cachedir in [config.clang_module_cache, config.lldb_module_cache]:
  if os.path.isdir(cachedir):
    print("Deleting module cache at %s." % cachedir)
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
dotest_cmd += ['--arch', config.test_arch]
dotest_cmd.extend(config.dotest_args_str.split(';'))

# Library path may be needed to locate just-built clang.
if config.llvm_libs_dir:
  dotest_cmd += ['--env', 'LLVM_LIBS_DIR=' + config.llvm_libs_dir]

# Forward ASan-specific environment variables to tests, as a test may load an
# ASan-ified dylib.
for env_var in ('ASAN_OPTIONS', 'DYLD_INSERT_LIBRARIES'):
  if env_var in config.environment:
    dotest_cmd += ['--inferior-env', env_var + '=' + config.environment[env_var]]

if config.lldb_build_directory:
  dotest_cmd += ['--build-dir', config.lldb_build_directory]

if config.lldb_module_cache:
  dotest_cmd += ['--lldb-module-cache-dir', config.lldb_module_cache]

if config.clang_module_cache:
  dotest_cmd += ['--clang-module-cache-dir', config.clang_module_cache]

if config.lldb_executable:
  dotest_cmd += ['--executable', config.lldb_executable]

if config.test_compiler:
  dotest_cmd += ['--compiler', config.test_compiler]

if config.dsymutil:
  dotest_cmd += ['--dsymutil', config.dsymutil]

if config.filecheck:
  dotest_cmd += ['--filecheck', config.filecheck]

if config.yaml2obj:
  dotest_cmd += ['--yaml2obj', config.yaml2obj]

if config.lldb_libs_dir:
  dotest_cmd += ['--lldb-libs-dir', config.lldb_libs_dir]

if 'lldb-repro-capture' in config.available_features or \
    'lldb-repro-replay' in config.available_features:
  dotest_cmd += ['--skip-category=lldb-vscode', '--skip-category=std-module']

if config.enabled_plugins:
  for plugin in config.enabled_plugins:
    dotest_cmd += ['--enable-plugin', plugin]

# We don't want to force users passing arguments to lit to use `;` as a
# separator. We use Python's simple lexical analyzer to turn the args into a
# list. Pass there arguments last so they can override anything that was
# already configured.
if config.dotest_lit_args_str:
  dotest_cmd.extend(shlex.split(config.dotest_lit_args_str))


# Load LLDB test format.
sys.path.append(os.path.join(config.lldb_src_root, "test", "API"))
import lldbtest

# testFormat: The test format to use to interpret tests.
config.test_format = lldbtest.LLDBTest(dotest_cmd)
