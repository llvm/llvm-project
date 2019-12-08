# -*- Python -*-

import os
import platform
import re
import shutil
import site
import subprocess
import sys

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst
from distutils.spawn import find_executable

site.addsitedir(os.path.dirname(__file__))
from helper import toolchain

# name: The name of this test suite.
config.name = 'lldb-shell'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = ['.test', '.cpp', '.s']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lldb_obj_root, 'test')

# Propagate LLDB_CAPTURE_REPRODUCER
if 'LLDB_CAPTURE_REPRODUCER' in os.environ:
  config.environment['LLDB_CAPTURE_REPRODUCER'] = os.environ[
      'LLDB_CAPTURE_REPRODUCER']

llvm_config.use_default_substitutions()
toolchain.use_lldb_substitutions(config)
toolchain.use_support_substitutions(config)


if re.match(r'^arm(hf.*-linux)|(.*-linux-gnuabihf)', config.target_triple):
    config.available_features.add("armhf-linux")

def calculate_arch_features(arch_string):
    # This will add a feature such as x86, arm, mips, etc for each built
    # target
    features = []
    for arch in arch_string.split():
        features.append(arch.lower())
    return features

# Run llvm-config and add automatically add features for whether we have
# assertions enabled, whether we are in debug mode, and what targets we
# are built for.
llvm_config.feature_config(
    [('--assertion-mode', {'ON': 'asserts'}),
     ('--build-mode', {'DEBUG': 'debug'}),
     ('--targets-built', calculate_arch_features)
     ])

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


# If running tests natively, check for CPU features needed for some tests.

if 'native' in config.available_features:
    cpuid_exe = lit.util.which('lit-cpuid', config.lldb_tools_dir)
    if cpuid_exe is None:
        lit_config.warning("lit-cpuid not found, tests requiring CPU extensions will be skipped")
    else:
        out, err, exitcode = lit.util.executeCommand([cpuid_exe])
        if exitcode == 0:
            for x in out.split():
                config.available_features.add('native-cpu-%s' % x)
        else:
            lit_config.warning("lit-cpuid failed: %s" % err)

if config.lldb_enable_python:
    config.available_features.add('python')

if config.lldb_enable_lua:
    config.available_features.add('lua')

if config.lldb_enable_lzma:
    config.available_features.add('lzma')

if find_executable('xz') != None:
    config.available_features.add('xz')

# NetBSD permits setting dbregs either if one is root
# or if user_set_dbregs is enabled
can_set_dbregs = True
if platform.system() == 'NetBSD' and os.geteuid() != 0:
    try:
        output = subprocess.check_output(["/sbin/sysctl", "-n",
          "security.models.extensions.user_set_dbregs"]).decode().strip()
        if output != "1":
            can_set_dbregs = False
    except subprocess.CalledProcessError:
        can_set_dbregs = False
if can_set_dbregs:
    config.available_features.add('dbregs-set')
