# -*- Python -*-

import json
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

site.addsitedir(os.path.dirname(__file__))
from helper import toolchain

# name: The name of this test suite.
config.name = "lldb-shell"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".test", ".cpp", ".s", ".m"]

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lldb_obj_root, "test", "Shell")

# Propagate environment vars.
llvm_config.with_system_environment(
    [
        "FREEBSD_LEGACY_PLUGIN",
        "HOME",
        "TEMP",
        "TMP",
        "XDG_CACHE_HOME",
    ]
)

# Enable sanitizer runtime flags.
if "Address" in config.llvm_use_sanitizer:
    config.environment["ASAN_OPTIONS"] = "detect_stack_use_after_return=1"
    if platform.system() == "Darwin":
        config.environment["MallocNanoZone"] = "0"

if "Thread" in config.llvm_use_sanitizer:
    config.environment["TSAN_OPTIONS"] = "halt_on_error=1"


# Support running the test suite under the lldb-repro wrapper. This makes it
# possible to capture a test suite run and then rerun all the test from the
# just captured reproducer.
lldb_repro_mode = lit_config.params.get("lldb-run-with-repro", None)
if lldb_repro_mode:
    config.available_features.add("lldb-repro")
    lit_config.note("Running Shell tests in {} mode.".format(lldb_repro_mode))
    toolchain.use_lldb_repro_substitutions(config, lldb_repro_mode)

llvm_config.use_default_substitutions()
toolchain.use_lldb_substitutions(config)
toolchain.use_support_substitutions(config)

if re.match(r"^arm(hf.*-linux)|(.*-linux-gnuabihf)", config.target_triple):
    config.available_features.add("armhf-linux")

if re.match(r".*-(windows-msvc)$", config.target_triple):
    config.available_features.add("windows-msvc")

if re.match(r".*-(windows-gnu|mingw32)$", config.target_triple):
    config.available_features.add("windows-gnu")


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
    [
        ("--assertion-mode", {"ON": "asserts"}),
        ("--build-mode", {"DEBUG": "debug"}),
        ("--targets-built", calculate_arch_features),
    ]
)

# Clean the module caches in the test build directory. This is necessary in an
# incremental build whenever clang changes underneath, so doing it once per
# lit.py invocation is close enough.
for cachedir in [config.clang_module_cache, config.lldb_module_cache]:
    if os.path.isdir(cachedir):
        lit_config.note("Deleting module cache at %s." % cachedir)
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

if "native" in config.available_features:
    cpuid_exe = lit.util.which("lit-cpuid", config.lldb_tools_dir)
    if cpuid_exe is None:
        lit_config.warning(
            "lit-cpuid not found, tests requiring CPU extensions will be skipped"
        )
    else:
        out, err, exitcode = lit.util.executeCommand([cpuid_exe])
        if exitcode == 0:
            for x in out.split():
                config.available_features.add("native-cpu-%s" % x)
        else:
            lit_config.warning("lit-cpuid failed: %s" % err)

if config.lldb_enable_python:
    config.available_features.add("python")

if config.lldb_enable_lua:
    config.available_features.add("lua")

if config.lldb_enable_lzma:
    config.available_features.add("lzma")

if shutil.which("xz") != None:
    config.available_features.add("xz")

if config.lldb_system_debugserver:
    config.available_features.add("system-debugserver")

if config.have_lldb_server:
    config.available_features.add("lldb-server")

if config.objc_gnustep_dir:
    config.available_features.add("objc-gnustep")
    if platform.system() == "Windows":
        # objc.dll must be in PATH since Windows has no rpath
        config.environment["PATH"] = os.path.pathsep.join(
            (
                os.path.join(config.objc_gnustep_dir, "lib"),
                config.environment.get("PATH", ""),
            )
        )

# NetBSD permits setting dbregs either if one is root
# or if user_set_dbregs is enabled
can_set_dbregs = True
if platform.system() == "NetBSD" and os.geteuid() != 0:
    try:
        output = (
            subprocess.check_output(
                ["/sbin/sysctl", "-n", "security.models.extensions.user_set_dbregs"]
            )
            .decode()
            .strip()
        )
        if output != "1":
            can_set_dbregs = False
    except subprocess.CalledProcessError:
        can_set_dbregs = False
if can_set_dbregs:
    config.available_features.add("dbregs-set")

if "LD_PRELOAD" in os.environ:
    config.available_features.add("ld_preload-present")

# Determine if a specific version of Xcode's linker contains a bug. We want to
# skip affected tests if they contain this bug.
if platform.system() == "Darwin":
    try:
        raw_version_details = subprocess.check_output(
            ("xcrun", "ld", "-version_details")
        )
        version_details = json.loads(raw_version_details)
        version = version_details.get("version", "0")
        version_tuple = tuple(int(x) for x in version.split("."))
        if (1000,) <= version_tuple <= (1109,):
            config.available_features.add("ld_new-bug")
    except:
        pass
