# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import sys
import re
import platform
import subprocess

import lit.util
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = "LLVM"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".ll", ".c", ".test", ".txt", ".s", ".mir", ".yaml"]

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.llvm_obj_root, "test")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# Propagate some variables from the host environment.
llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])


# Set up OCAMLPATH to include newly built OCaml libraries.
top_ocaml_lib = os.path.join(config.llvm_lib_dir, "ocaml")
llvm_ocaml_lib = os.path.join(top_ocaml_lib, "llvm")

llvm_config.with_system_environment("OCAMLPATH")
llvm_config.with_environment("OCAMLPATH", top_ocaml_lib, append_path=True)
llvm_config.with_environment("OCAMLPATH", llvm_ocaml_lib, append_path=True)

llvm_config.with_system_environment("CAML_LD_LIBRARY_PATH")
llvm_config.with_environment("CAML_LD_LIBRARY_PATH", llvm_ocaml_lib, append_path=True)

# Set up OCAMLRUNPARAM to enable backtraces in OCaml tests.
llvm_config.with_environment("OCAMLRUNPARAM", "b")

# Provide the path to asan runtime lib 'libclang_rt.asan_osx_dynamic.dylib' if
# available. This is darwin specific since it's currently only needed on darwin.


def get_asan_rtlib():
    if (
        not "Address" in config.llvm_use_sanitizer
        or not "Darwin" in config.host_os
        or not "x86" in config.host_triple
    ):
        return ""
    try:
        import glob
    except:
        print("glob module not found, skipping get_asan_rtlib() lookup")
        return ""
    # The libclang_rt.asan_osx_dynamic.dylib path is obtained using the relative
    # path from the host cc.
    host_lib_dir = os.path.join(os.path.dirname(config.host_cc), "../lib")
    asan_dylib_dir_pattern = (
        host_lib_dir + "/clang/*/lib/darwin/libclang_rt.asan_osx_dynamic.dylib"
    )
    found_dylibs = glob.glob(asan_dylib_dir_pattern)
    if len(found_dylibs) != 1:
        return ""
    return found_dylibs[0]


llvm_config.use_default_substitutions()

# Add site-specific substitutions.
config.substitutions.append(("%llvmshlibdir", config.llvm_shlib_dir))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%pluginext", config.llvm_plugin_ext))
config.substitutions.append(("%exeext", config.llvm_exe_ext))


lli_args = []
# The target triple used by default by lli is the process target triple (some
# triple appropriate for generating code for the current process) but because
# we don't support COFF in MCJIT well enough for the tests, force ELF format on
# Windows.  FIXME: the process target triple should be used here, but this is
# difficult to obtain on Windows.
if re.search(r"cygwin|windows-gnu|windows-msvc", config.host_triple):
    lli_args = ["-mtriple=" + config.host_triple + "-elf"]

llc_args = []

# Similarly, have a macro to use llc with DWARF even when the host is Windows
if re.search(r"windows-msvc", config.target_triple):
    llc_args = [" -mtriple=" + config.target_triple.replace("-msvc", "-gnu")]

# Provide the path to asan runtime lib if available. On darwin, this lib needs
# to be loaded via DYLD_INSERT_LIBRARIES before libLTO.dylib in case the files
# to be linked contain instrumented sanitizer code.
ld64_cmd = config.ld64_executable
asan_rtlib = get_asan_rtlib()
if asan_rtlib:
    ld64_cmd = "DYLD_INSERT_LIBRARIES={} {}".format(asan_rtlib, ld64_cmd)
if config.osx_sysroot:
    ld64_cmd = "{} -syslibroot {}".format(ld64_cmd, config.osx_sysroot)

ocamlc_command = "%s ocamlc -cclib -L%s %s" % (
    config.ocamlfind_executable,
    config.llvm_lib_dir,
    config.ocaml_flags,
)
ocamlopt_command = "true"
if config.have_ocamlopt:
    ocamlopt_command = "%s ocamlopt -cclib -L%s -cclib -Wl,-rpath,%s %s" % (
        config.ocamlfind_executable,
        config.llvm_lib_dir,
        config.llvm_lib_dir,
        config.ocaml_flags,
    )

opt_viewer_cmd = "%s %s/tools/opt-viewer/opt-viewer.py" % (
    sys.executable,
    config.llvm_src_root,
)

llvm_original_di_preservation_cmd = os.path.join(
    config.llvm_src_root, "utils", "llvm-original-di-preservation.py"
)
config.substitutions.append(
    (
        "%llvm-original-di-preservation",
        "'%s' %s" % (config.python_executable, llvm_original_di_preservation_cmd),
    )
)

llvm_locstats_tool = os.path.join(config.llvm_tools_dir, "llvm-locstats")
config.substitutions.append(
    ("%llvm-locstats", "'%s' %s" % (config.python_executable, llvm_locstats_tool))
)
config.llvm_locstats_used = os.path.exists(llvm_locstats_tool)

tools = [
    ToolSubst("%llvm", FindTool("llvm"), unresolved="ignore"),
    ToolSubst("%lli", FindTool("lli"), post=".", extra_args=lli_args),
    ToolSubst("%llc_dwarf", FindTool("llc"), extra_args=llc_args),
    ToolSubst("%gold", config.gold_executable, unresolved="ignore"),
    ToolSubst("%ld64", ld64_cmd, unresolved="ignore"),
    ToolSubst("%ocamlc", ocamlc_command, unresolved="ignore"),
    ToolSubst("%ocamlopt", ocamlopt_command, unresolved="ignore"),
    ToolSubst("%opt-viewer", opt_viewer_cmd),
    ToolSubst("%llvm-objcopy", FindTool("llvm-objcopy")),
    ToolSubst("%llvm-strip", FindTool("llvm-strip")),
    ToolSubst("%llvm-install-name-tool", FindTool("llvm-install-name-tool")),
    ToolSubst("%llvm-bitcode-strip", FindTool("llvm-bitcode-strip")),
    ToolSubst("%split-file", FindTool("split-file")),
]

# FIXME: Why do we have both `lli` and `%lli` that do slightly different things?
tools.extend(
    [
        "dsymutil",
        "lli",
        "lli-child-target",
        "llvm-ar",
        "llvm-as",
        "llvm-addr2line",
        "llvm-bcanalyzer",
        "llvm-bitcode-strip",
        "llvm-config",
        "llvm-cov",
        "llvm-cxxdump",
        "llvm-cvtres",
        "llvm-debuginfod-find",
        "llvm-debuginfo-analyzer",
        "llvm-diff",
        "llvm-dis",
        "llvm-dwarfdump",
        "llvm-dwarfutil",
        "llvm-dwp",
        "llvm-dlltool",
        "llvm-exegesis",
        "llvm-extract",
        "llvm-isel-fuzzer",
        "llvm-ifs",
        "llvm-install-name-tool",
        "llvm-jitlink",
        "llvm-opt-fuzzer",
        "llvm-lib",
        "llvm-link",
        "llvm-lto",
        "llvm-lto2",
        "llvm-mc",
        "llvm-mca",
        "llvm-modextract",
        "llvm-nm",
        "llvm-objcopy",
        "llvm-objdump",
        "llvm-otool",
        "llvm-pdbutil",
        "llvm-profdata",
        "llvm-profgen",
        "llvm-ranlib",
        "llvm-rc",
        "llvm-readelf",
        "llvm-readobj",
        "llvm-rtdyld",
        "llvm-sim",
        "llvm-size",
        "llvm-split",
        "llvm-stress",
        "llvm-strings",
        "llvm-strip",
        "llvm-tblgen",
        "llvm-readtapi",
        "llvm-undname",
        "llvm-windres",
        "llvm-c-test",
        "llvm-cxxfilt",
        "llvm-xray",
        "yaml2obj",
        "obj2yaml",
        "yaml-bench",
        "verify-uselistorder",
        "bugpoint",
        "llc",
        "llvm-symbolizer",
        "opt",
        "sancov",
        "sanstats",
        "llvm-remarkutil",
    ]
)

# The following tools are optional
tools.extend(
    [
        ToolSubst("llvm-mt", unresolved="ignore"),
        ToolSubst("llvm-debuginfod", unresolved="ignore"),
        ToolSubst("Kaleidoscope-Ch3", unresolved="ignore"),
        ToolSubst("Kaleidoscope-Ch4", unresolved="ignore"),
        ToolSubst("Kaleidoscope-Ch5", unresolved="ignore"),
        ToolSubst("Kaleidoscope-Ch6", unresolved="ignore"),
        ToolSubst("Kaleidoscope-Ch7", unresolved="ignore"),
        ToolSubst("Kaleidoscope-Ch8", unresolved="ignore"),
        ToolSubst("LLJITWithThinLTOSummaries", unresolved="ignore"),
        ToolSubst("LLJITWithRemoteDebugging", unresolved="ignore"),
        ToolSubst("OrcV2CBindingsBasicUsage", unresolved="ignore"),
        ToolSubst("OrcV2CBindingsAddObjectFile", unresolved="ignore"),
        ToolSubst("OrcV2CBindingsRemovableCode", unresolved="ignore"),
        ToolSubst("OrcV2CBindingsLazy", unresolved="ignore"),
        ToolSubst("OrcV2CBindingsVeryLazy", unresolved="ignore"),
        ToolSubst("dxil-dis", unresolved="ignore"),
    ]
)

# Find (major, minor) version of ptxas
def ptxas_version(ptxas):
    ptxas_cmd = subprocess.Popen([ptxas, "--version"], stdout=subprocess.PIPE)
    ptxas_out = ptxas_cmd.stdout.read().decode("ascii")
    ptxas_cmd.wait()
    match = re.search(r"release (\d+)\.(\d+)", ptxas_out)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    print("couldn't determine ptxas version")
    return None


# Enable %ptxas and %ptxas-verify tools.
# %ptxas-verify defaults to sm_60 architecture. It can be overriden
# by specifying required one, for instance: %ptxas-verify -arch=sm_80.
def enable_ptxas(ptxas_executable):
    version = ptxas_version(ptxas_executable)
    if version:
        # ptxas is supposed to be backward compatible with previous
        # versions, so add a feature for every known version prior to
        # the current one.
        ptxas_known_versions = [
            (9, 0),
            (9, 1),
            (9, 2),
            (10, 0),
            (10, 1),
            (10, 2),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (12, 0),
            (12, 1),
        ]

        def version_int(ver):
            return ver[0] * 100 + ver[1]

        # ignore ptxas if its version is below the minimum supported
        # version
        min_version = ptxas_known_versions[0]
        if version_int(version) < version_int(min_version):
            print(
                "Warning: ptxas version {}.{} is not supported".format(
                    version[0], version[1]
                )
            )
            return

        for known_version in ptxas_known_versions:
            if version_int(known_version) <= version_int(version):
                major, minor = known_version
                config.available_features.add("ptxas-{}.{}".format(major, minor))

    config.available_features.add("ptxas")
    tools.extend(
        [
            ToolSubst("%ptxas", ptxas_executable),
            ToolSubst("%ptxas-verify", "{} -arch=sm_60 -c -".format(ptxas_executable)),
        ]
    )


ptxas_executable = (
    os.environ.get("LLVM_PTXAS_EXECUTABLE", None) or config.ptxas_executable
)
if ptxas_executable:
    enable_ptxas(ptxas_executable)

llvm_config.add_tool_substitutions(tools, config.llvm_tools_dir)

# Targets

config.targets = frozenset(config.targets_to_build.split())

for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + "-registered-target")

# Features
known_arches = ["x86_64", "mips64", "ppc64", "aarch64"]
if config.host_ldflags.find("-m32") < 0 and any(
    config.llvm_host_triple.startswith(x) for x in known_arches
):
    config.available_features.add("llvm-64-bits")

config.available_features.add("host-byteorder-" + sys.byteorder + "-endian")

if sys.platform in ["win32"]:
    # ExecutionEngine, no weak symbols in COFF.
    config.available_features.add("uses_COFF")
else:
    # Others/can-execute.txt
    config.available_features.add("can-execute")

# Loadable module
if config.has_plugins:
    config.available_features.add("plugins")

if config.build_examples:
    config.available_features.add("examples")

if config.linked_bye_extension:
    config.substitutions.append(("%llvmcheckext", "CHECK-EXT"))
    config.substitutions.append(("%loadbye", ""))
    config.substitutions.append(("%loadnewpmbye", ""))
else:
    config.substitutions.append(("%llvmcheckext", "CHECK-NOEXT"))
    config.substitutions.append(
        (
            "%loadbye",
            "-load={}/Bye{}".format(config.llvm_shlib_dir, config.llvm_shlib_ext),
        )
    )
    config.substitutions.append(
        (
            "%loadnewpmbye",
            "-load-pass-plugin={}/Bye{}".format(
                config.llvm_shlib_dir, config.llvm_shlib_ext
            ),
        )
    )

if config.linked_exampleirtransforms_extension:
    config.substitutions.append(("%loadexampleirtransforms", ""))
else:
    config.substitutions.append(
        (
            "%loadexampleirtransforms",
            "-load-pass-plugin={}/ExampleIRTransforms{}".format(
                config.llvm_shlib_dir, config.llvm_shlib_ext
            ),
        )
    )

# Static libraries are not built if BUILD_SHARED_LIBS is ON.
if not config.build_shared_libs and not config.link_llvm_dylib:
    config.available_features.add("static-libs")

if config.link_llvm_dylib:
    config.available_features.add("llvm-dylib")
    config.substitutions.append(
        (
            # libLLVM.so.19.0git
            "%llvmdylib",
            "{}/libLLVM{}.{}".format(
                config.llvm_shlib_dir, config.llvm_shlib_ext, config.llvm_dylib_version
            )
        )
    )

if config.have_tf_aot:
    config.available_features.add("have_tf_aot")

if config.have_tflite:
    config.available_features.add("have_tflite")

if config.llvm_inliner_model_autogenerated:
    config.available_features.add("llvm_inliner_model_autogenerated")

if config.llvm_raevict_model_autogenerated:
    config.available_features.add("llvm_raevict_model_autogenerated")


def have_cxx_shared_library():
    readobj_exe = lit.util.which("llvm-readobj", config.llvm_tools_dir)
    if not readobj_exe:
        print("llvm-readobj not found")
        return False

    try:
        readobj_cmd = subprocess.Popen(
            [readobj_exe, "--needed-libs", readobj_exe], stdout=subprocess.PIPE
        )
    except OSError:
        print("could not exec llvm-readobj")
        return False

    readobj_out = readobj_cmd.stdout.read().decode("ascii")
    readobj_cmd.wait()

    regex = re.compile(r"(libc\+\+|libstdc\+\+|msvcp).*\.(so|dylib|dll)")
    needed_libs = False
    for line in readobj_out.splitlines():
        if "NeededLibraries [" in line:
            needed_libs = True
        if "]" in line:
            needed_libs = False
        if needed_libs and regex.search(line.lower()):
            return True
    return False


if have_cxx_shared_library():
    config.available_features.add("cxx-shared-library")

if config.libcxx_used:
    config.available_features.add("libcxx-used")

# LLVM can be configured with an empty default triple
# Some tests are "generic" and require a valid default triple
if config.target_triple:
    config.available_features.add("default_triple")
    # Direct object generation
    if not config.target_triple.startswith(("nvptx", "xcore")):
        config.available_features.add("object-emission")

if config.have_llvm_driver:
    config.available_features.add("llvm-driver")

import subprocess


def have_ld_plugin_support():
    if not os.path.exists(
        os.path.join(config.llvm_shlib_dir, "LLVMgold" + config.llvm_shlib_ext)
    ):
        return False

    ld_cmd = subprocess.Popen(
        [config.gold_executable, "--help"], stdout=subprocess.PIPE, env={"LANG": "C"}
    )
    ld_out = ld_cmd.stdout.read().decode()
    ld_cmd.wait()

    if not "-plugin" in ld_out:
        return False

    # check that the used emulations are supported.
    emu_line = [l for l in ld_out.split("\n") if "supported emulations" in l]
    if len(emu_line) != 1:
        return False
    emu_line = emu_line[0]
    fields = emu_line.split(":")
    if len(fields) != 3:
        return False
    emulations = fields[2].split()
    if "elf_x86_64" not in emulations:
        return False
    if "elf32ppc" in emulations:
        config.available_features.add("ld_emu_elf32ppc")

    ld_version = subprocess.Popen(
        [config.gold_executable, "--version"], stdout=subprocess.PIPE, env={"LANG": "C"}
    )
    if not "GNU gold" in ld_version.stdout.read().decode():
        return False
    ld_version.wait()

    return True


if have_ld_plugin_support():
    config.available_features.add("ld_plugin")


def have_ld64_plugin_support():
    if not os.path.exists(
        os.path.join(config.llvm_shlib_dir, "libLTO" + config.llvm_shlib_ext)
    ):
        return False

    if config.ld64_executable == "":
        return False

    ld_cmd = subprocess.Popen([config.ld64_executable, "-v"], stderr=subprocess.PIPE)
    ld_out = ld_cmd.stderr.read().decode()
    ld_cmd.wait()

    if "ld64" not in ld_out or "LTO" not in ld_out:
        return False

    return True


if have_ld64_plugin_support():
    config.available_features.add("ld64_plugin")

# Ask llvm-config about asserts
llvm_config.feature_config(
    [
        ("--assertion-mode", {"ON": "asserts"}),
        ("--build-mode", {"[Dd][Ee][Bb][Uu][Gg]": "debug"}),
    ]
)

if "darwin" == sys.platform:
    cmd = ["sysctl", "hw.optional.fma"]
    sysctl_cmd = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    # Non zero return, probably a permission issue
    if sysctl_cmd.wait():
        print(
            'Warning: sysctl exists but calling "{}" failed, defaulting to no fma3.'.format(
                " ".join(cmd)
            )
        )
    else:
        result = sysctl_cmd.stdout.read().decode("ascii")
        if "hw.optional.fma: 1" in result:
            config.available_features.add("fma3")

if not hasattr(sys, "getwindowsversion") or sys.getwindowsversion().build >= 17063:
    config.available_features.add("unix-sockets")

# .debug_frame is not emitted for targeting Windows x64, aarch64/arm64, AIX, or Apple Silicon Mac.
if not re.match(
    r"^(x86_64|aarch64|arm64|powerpc|powerpc64).*-(windows-gnu|windows-msvc|aix)",
    config.target_triple,
) and not re.match(r"^arm64(e)?-apple-(macos|darwin)", config.target_triple):
    config.available_features.add("debug_frame")

if config.enable_threads:
    config.available_features.add("thread_support")

if config.have_libxml2:
    config.available_features.add("libxml2")

if config.have_curl:
    config.available_features.add("curl")

if config.have_httplib:
    config.available_features.add("httplib")

if config.have_opt_viewer_modules:
    config.available_features.add("have_opt_viewer_modules")

if config.expensive_checks:
    config.available_features.add("expensive_checks")

if "MemoryWithOrigins" in config.llvm_use_sanitizer:
    config.available_features.add("use_msan_with_origins")


# Some tools support an environment variable "OBJECT_MODE" on AIX OS, which
# controls the kind of objects they will support. If there is no "OBJECT_MODE"
# environment variable specified, the default behaviour is to support 32-bit
# objects only. In order to not affect most test cases, which expect to support
# 32-bit and 64-bit objects by default, set the environment variable
# "OBJECT_MODE" to 'any' by default on AIX OS.
if "system-aix" in config.available_features:
    config.environment["OBJECT_MODE"] = "any"
