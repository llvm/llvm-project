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

config.test_format = toolchain.ShTestLldb()

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".test", ".cpp", ".s", ".m", ".ll", ".c"]

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
if config.llvm_use_sanitizer:
    config.environment["ASAN_OPTIONS"] = "detect_stack_use_after_return=1"
    config.environment["TSAN_OPTIONS"] = "halt_on_error=1"
    config.environment["MallocNanoZone"] = "0"

if config.lldb_platform_url and config.cmake_sysroot and config.enable_remote:
    if re.match(r".*-linux.*", config.target_triple):
        config.available_features.add("remote-linux")
else:
    # After this, enable_remote == True iff remote testing is going to be used.
    config.enable_remote = False

llvm_config.use_default_substitutions()
toolchain.use_lldb_substitutions(config)
toolchain.use_support_substitutions(config)

if re.match(r"^arm(hf.*-linux)|(.*-linux-gnuabihf)", config.target_triple):
    config.available_features.add("armhf-linux")

if re.match(r".*-(windows|mingw32)", config.target_triple):
    config.available_features.add("target-windows")

if re.match(r".*-(windows-msvc)$", config.target_triple):
    config.available_features.add("windows-msvc")

if re.match(r".*-(windows-gnu|mingw32)$", config.target_triple):
    config.available_features.add("windows-gnu")

if config.targets_to_build:
    for arch in config.targets_to_build.split(";"):
        if arch:
            config.available_features.add(arch.lower() + "-registered-target")

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
    config.maxIndividualTestTime = 600
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

if getattr(config, "lldb_enable_mte", False):
    config.available_features.add("lldb-mte")

if config.lldb_enable_lua:
    config.available_features.add("lua")

if config.lldb_enable_lzma:
    config.available_features.add("lzma")

if shutil.which("xz") is not None:
    config.available_features.add("xz")

if config.lldb_system_debugserver:
    config.available_features.add("system-debugserver")

if config.have_lldb_server:
    config.available_features.add("lldb-server")


# Same spelling as lldb/test/Shell/CMakeLists.txt and lldb/test/API/lit.cfg.py.
def targets_mingw(triple):
    return re.search(r"windows-gnu|mingw", triple or "") is not None


def runtime_exports(directory, symbol):
    """Whether a libobjc2 under `directory` exports `symbol`."""

    def tool(name):
        return shutil.which(name, path=config.llvm_tools_dir) or shutil.which(name)

    for subdir in ("lib", "bin"):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for entry in os.listdir(subdir_path):
            root, ext = os.path.splitext(entry)
            is_pe = ext.lower() == ".dll"
            is_unix_shared = ext.lower() in (".so", ".dylib") or ".so." in entry
            if not (is_pe or is_unix_shared):
                continue
            if not re.match(r"(lib)?objc([.-]|$)", root, re.IGNORECASE):
                continue
            path = os.path.join(subdir_path, entry)
            # The export/dynamic table, not the symbol table: a stripped
            # library still exports.
            if is_pe:
                argv = [tool("llvm-readobj"), "--coff-exports", path]
            else:
                argv = [tool("llvm-nm"), "--dynamic", "--defined-only", path]
            if not argv[0]:
                lit_config.warning("no tool to read exports from " + path)
                continue
            try:
                probe = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as e:
                lit_config.warning("could not read exports from %s: %s" % (path, e))
                continue
            if probe.returncode != 0:
                lit_config.warning("could not read exports from " + path)
                continue
            if symbol in probe.stdout:
                return True
    return False


# A test whose inferior is built for another target says so itself, rather
# than the suite claiming it for every invocation: most of the suite debugs
# MSVC binaries, which such a claim would describe wrongly. Both expand to
# nothing when inferiors are built for the host, so nothing else moves.
#
# Underscores, not hyphens: ToolSubst wraps a key in , so %inferior-abi
# would be eaten by an existing %inferior substitution if one were ever added.
_test_triple = getattr(config, "test_triple", None)
config.substitutions.append(
    (
        "%inferior_abi",
        # -O, not -o: the setting only takes effect before a target exists.
        (
            '-O "settings set plugin.object-file.pe-coff.abi gnu"'
            if targets_mingw(_test_triple)
            else ""
        ),
    )
)
config.substitutions.append(
    (
        "%inferior_target",
        (
            "--target=" + _test_triple
            if _test_triple and _test_triple != config.target_triple
            else ""
        ),
    )
)

# Windows has no rpath, so a MinGW-built inferior needs its own toolchain's
# DLLs (libstdc++, libgcc) ahead of any other distribution's on PATH.
if (
    platform.system() == "Windows"
    and config.cmake_sysroot
    and targets_mingw(getattr(config, "test_triple", None))
):
    config.environment["PATH"] = os.path.pathsep.join(
        (
            os.path.join(config.cmake_sysroot, "bin"),
            config.environment.get("PATH", ""),
        )
    )

if config.objc_gnustep_dir:
    config.available_features.add("objc-gnustep")
    if platform.system() == "Windows":
        # No rpath on Windows. MSVC libobjc2 installs the DLL in lib/, MinGW
        # in bin/.
        config.environment["PATH"] = os.path.pathsep.join(
            (
                os.path.join(config.objc_gnustep_dir, "lib"),
                os.path.join(config.objc_gnustep_dir, "bin"),
                config.environment.get("PATH", ""),
            )
        )

    # Catch breakpoints need an entry point for entering a handler. libobjc2
    # exports one only where exceptions unwind through the Itanium ABI: a
    # property of how the configured runtime was built, not of any triple.
    if runtime_exports(config.objc_gnustep_dir, "objc_begin_catch"):
        config.available_features.add("objc-gnustep-catch")

if config.have_dia_sdk:
    config.available_features.add("diasdk")

if platform.system() == "Windows":
    config.environment["LLDB_USE_LLDB_SERVER"] = (
        "1" if getattr(config, "lldb_use_lldb_server", False) else "0"
    )
    # Use anonymous pipes instead of ConPTY for all tests. ConPTY injects VT
    # escape sequences into the output stream, which breaks tests that check
    # for specific stdout/stderr content.
    config.environment["LLDB_LAUNCH_FLAG_USE_PIPES"] = "1"

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

# Some shell tests dynamically link with python.dll and need to know the
# location of the Python libraries. This ensures that we use the same
# version of Python that was used to build lldb to run our tests.
config.environment["PYTHONHOME"] = config.python_root_dir
config.environment["PATH"] = os.path.pathsep.join(
    (config.python_root_dir, config.environment.get("PATH", ""))
)
