# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import platform
import shlex
import shutil
import subprocess

import lit.formats

# name: The name of this test suite.
config.name = 'lldb-api'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lldb_obj_root, 'test', 'API')

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


def is_configured(attr):
  """Return the configuration attribute if it exists and None otherwise.

  This allows us to check if the attribute exists before trying to access it."""
  return getattr(config, attr, None)


def delete_module_cache(path):
  """Clean the module caches in the test build directory.

  This is necessary in an incremental build whenever clang changes underneath,
  so doing it once per lit.py invocation is close enough. """
  if os.path.isdir(path):
    lit_config.note("Deleting module cache at %s." % path)
    shutil.rmtree(path)

if is_configured('llvm_use_sanitizer'):
  if 'Address' in config.llvm_use_sanitizer:
    config.environment['ASAN_OPTIONS'] = 'detect_stack_use_after_return=1'
    if 'Darwin' in config.host_os:
      config.environment['DYLD_INSERT_LIBRARIES'] = find_sanitizer_runtime(
          'libclang_rt.asan_osx_dynamic.dylib')

  if 'Thread' in config.llvm_use_sanitizer:
    if 'Darwin' in config.host_os:
      config.environment['DYLD_INSERT_LIBRARIES'] = find_sanitizer_runtime(
          'libclang_rt.tsan_osx_dynamic.dylib')

if 'DYLD_INSERT_LIBRARIES' in config.environment and platform.system() == 'Darwin':
  config.python_executable = find_python_interpreter()

# Shared library build of LLVM may require LD_LIBRARY_PATH or equivalent.
if is_configured('shared_libs'):
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

lldb_use_simulator = lit_config.params.get('lldb-run-with-simulator', None)
if lldb_use_simulator:
  if lldb_use_simulator == "ios":
    lit_config.note("Running API tests on iOS simulator")
    config.available_features.add('lldb-simulator-ios')
  elif lldb_use_simulator == "watchos":
    lit_config.note("Running API tests on watchOS simulator")
    config.available_features.add('lldb-simulator-watchos')
  elif lldb_use_simulator == "tvos":
    lit_config.note("Running API tests on tvOS simulator")
    config.available_features.add('lldb-simulator-tvos')
  else:
    lit_config.error("Unknown simulator id '{}'".format(lldb_use_simulator))

# Set a default per-test timeout of 10 minutes. Setting a timeout per test
# requires that killProcessAndChildren() is supported on the platform and
# lit complains if the value is set but it is not supported.
supported, errormsg = lit_config.maxIndividualTestTimeIsSupported
if supported:
  lit_config.maxIndividualTestTime = 600
else:
  lit_config.warning("Could not set a default per-test timeout. " + errormsg)

# Build dotest command.
dotest_cmd = [os.path.join(config.lldb_src_root, 'test', 'API', 'dotest.py')]

if is_configured('dotest_common_args_str'):
  dotest_cmd.extend(config.dotest_common_args_str.split(';'))

# Library path may be needed to locate just-built clang and libcxx.
if is_configured('llvm_libs_dir'):
  dotest_cmd += ['--env', 'LLVM_LIBS_DIR=' + config.llvm_libs_dir]

# Include path may be needed to locate just-built libcxx.
if is_configured('llvm_include_dir'):
  dotest_cmd += ['--env', 'LLVM_INCLUDE_DIR=' + config.llvm_include_dir]

# This path may be needed to locate required llvm tools
if is_configured('llvm_tools_dir'):
  dotest_cmd += ['--env', 'LLVM_TOOLS_DIR=' + config.llvm_tools_dir]

# If we have a just-built libcxx, prefer it over the system one.
if is_configured('has_libcxx') and config.has_libcxx:
  if platform.system() != 'Windows':
    if is_configured('llvm_include_dir') and is_configured('llvm_libs_dir'):
      dotest_cmd += ['--libcxx-include-dir', os.path.join(config.llvm_include_dir, 'c++', 'v1')]
      dotest_cmd += ['--libcxx-library-dir', config.llvm_libs_dir]

# Forward ASan-specific environment variables to tests, as a test may load an
# ASan-ified dylib.
for env_var in ('ASAN_OPTIONS', 'DYLD_INSERT_LIBRARIES'):
  if env_var in config.environment:
    dotest_cmd += ['--inferior-env', env_var + '=' + config.environment[env_var]]

if is_configured('test_arch'):
  dotest_cmd += ['--arch', config.test_arch]

if is_configured('lldb_build_directory'):
  dotest_cmd += ['--build-dir', config.lldb_build_directory]

if is_configured('lldb_module_cache'):
  delete_module_cache(config.lldb_module_cache)
  dotest_cmd += ['--lldb-module-cache-dir', config.lldb_module_cache]

if is_configured('clang_module_cache'):
  delete_module_cache(config.clang_module_cache)
  dotest_cmd += ['--clang-module-cache-dir', config.clang_module_cache]

if is_configured('lldb_executable'):
  dotest_cmd += ['--executable', config.lldb_executable]

if is_configured('test_compiler'):
  dotest_cmd += ['--compiler', config.test_compiler]

if is_configured('dsymutil'):
  dotest_cmd += ['--dsymutil', config.dsymutil]

if is_configured('llvm_tools_dir'):
  dotest_cmd += ['--llvm-tools-dir', config.llvm_tools_dir]

if is_configured('server'):
  dotest_cmd += ['--server', config.server]

if is_configured('lldb_libs_dir'):
  dotest_cmd += ['--lldb-libs-dir', config.lldb_libs_dir]

if is_configured('lldb_framework_dir'):
  dotest_cmd += ['--framework', config.lldb_framework_dir]

if 'lldb-repro-capture' in config.available_features or \
    'lldb-repro-replay' in config.available_features:
  dotest_cmd += ['--skip-category=lldb-vscode', '--skip-category=std-module']

if 'lldb-simulator-ios' in config.available_features:
  dotest_cmd += ['--apple-sdk', 'iphonesimulator',
                 '--platform-name', 'ios-simulator']
elif 'lldb-simulator-watchos' in config.available_features:
  dotest_cmd += ['--apple-sdk', 'watchsimulator',
                 '--platform-name', 'watchos-simulator']
elif 'lldb-simulator-tvos' in config.available_features:
  dotest_cmd += ['--apple-sdk', 'appletvsimulator',
                 '--platform-name', 'tvos-simulator']

if is_configured('enabled_plugins'):
  for plugin in config.enabled_plugins:
    dotest_cmd += ['--enable-plugin', plugin]

# `dotest` args come from three different sources:
# 1. Derived by CMake based on its configs (LLDB_TEST_COMMON_ARGS), which end
# up in `dotest_common_args_str`.
# 2. CMake user parameters (LLDB_TEST_USER_ARGS), which end up in
# `dotest_user_args_str`.
# 3. With `llvm-lit "--param=dotest-args=..."`, which end up in
# `dotest_lit_args_str`.
# Check them in this order, so that more specific overrides are visited last.
# In particular, (1) is visited at the top of the file, since the script
# derives other information from it.

if is_configured('dotest_user_args_str'):
  dotest_cmd.extend(config.dotest_user_args_str.split(';'))

if is_configured('dotest_lit_args_str'):
  # We don't want to force users passing arguments to lit to use `;` as a
  # separator. We use Python's simple lexical analyzer to turn the args into a
  # list. Pass there arguments last so they can override anything that was
  # already configured.
  dotest_cmd.extend(shlex.split(config.dotest_lit_args_str))

# Load LLDB test format.
sys.path.append(os.path.join(config.lldb_src_root, "test", "API"))
import lldbtest

# testFormat: The test format to use to interpret tests.
config.test_format = lldbtest.LLDBTest(dotest_cmd)

# Propagate TERM or default to vt100.
config.environment['TERM'] = os.getenv('TERM', default='vt100')

# Propagate FREEBSD_LEGACY_PLUGIN
if 'FREEBSD_LEGACY_PLUGIN' in os.environ:
  config.environment['FREEBSD_LEGACY_PLUGIN'] = os.environ[
      'FREEBSD_LEGACY_PLUGIN']

# Propagate XDG_CACHE_HOME
if 'XDG_CACHE_HOME' in os.environ:
  config.environment['XDG_CACHE_HOME'] = os.environ['XDG_CACHE_HOME']
