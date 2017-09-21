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

# name: The name of this test suite.
config.name = 'LLVM'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = ['.ll', '.c', '.cxx', '.test', '.txt', '.s', '.mir']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.llvm_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# Propagate some variables from the host environment.
llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP', 'ASAN_SYMBOLIZER_PATH', 'MSAN_SYMBOLIZER_PATH'])


# Set up OCAMLPATH to include newly built OCaml libraries.
top_ocaml_lib = os.path.join(config.llvm_lib_dir, 'ocaml')
llvm_ocaml_lib = os.path.join(top_ocaml_lib, 'llvm')

llvm_config.with_system_environment('OCAMLPATH')
llvm_config.with_environment('OCAMLPATH', top_ocaml_lib, append_path=True)
llvm_config.with_environment('OCAMLPATH', llvm_ocaml_lib, append_path=True)

llvm_config.with_system_environment('CAML_LD_LIBRARY_PATH')
llvm_config.with_environment('CAML_LD_LIBRARY_PATH', llvm_ocaml_lib, append_path=True)

# Set up OCAMLRUNPARAM to enable backtraces in OCaml tests.
llvm_config.with_environment('OCAMLRUNPARAM', 'b')

# Provide the path to asan runtime lib 'libclang_rt.asan_osx_dynamic.dylib' if
# available. This is darwin specific since it's currently only needed on darwin.
def get_asan_rtlib():
    if not "Address" in config.llvm_use_sanitizer or \
       not "Darwin" in config.host_os or \
       not "x86" in config.host_triple:
        return ""
    try:
        import glob
    except:
        print("glob module not found, skipping get_asan_rtlib() lookup")
        return ""
    # The libclang_rt.asan_osx_dynamic.dylib path is obtained using the relative
    # path from the host cc.
    host_lib_dir = os.path.join(os.path.dirname(config.host_cc), "../lib")
    asan_dylib_dir_pattern = host_lib_dir + \
        "/clang/*/lib/darwin/libclang_rt.asan_osx_dynamic.dylib"
    found_dylibs = glob.glob(asan_dylib_dir_pattern)
    if len(found_dylibs) != 1:
        return ""
    return found_dylibs[0]

lli = 'lli'
# The target triple used by default by lli is the process target triple (some
# triple appropriate for generating code for the current process) but because
# we don't support COFF in MCJIT well enough for the tests, force ELF format on
# Windows.  FIXME: the process target triple should be used here, but this is
# difficult to obtain on Windows.
if re.search(r'cygwin|mingw32|windows-gnu|windows-msvc|win32', config.host_triple):
  lli += ' -mtriple='+config.host_triple+'-elf'
config.substitutions.append( ('%lli', lli ) )

# Similarly, have a macro to use llc with DWARF even when the host is win32.
llc_dwarf = 'llc'
if re.search(r'win32', config.target_triple):
  llc_dwarf += ' -mtriple='+config.target_triple.replace('-win32', '-mingw32')
config.substitutions.append( ('%llc_dwarf', llc_dwarf) )

# Add site-specific substitutions.
config.substitutions.append( ('%gold', config.gold_executable) )
config.substitutions.append( ('%go', config.go_executable) )
config.substitutions.append( ('%llvmshlibdir', config.llvm_shlib_dir) )
config.substitutions.append( ('%shlibext', config.llvm_shlib_ext) )
config.substitutions.append( ('%exeext', config.llvm_exe_ext) )
config.substitutions.append( ('%python', config.python_executable) )
config.substitutions.append( ('%host_cc', config.host_cc) )

# Provide the path to asan runtime lib if available. On darwin, this lib needs
# to be loaded via DYLD_INSERT_LIBRARIES before libLTO.dylib in case the files
# to be linked contain instrumented sanitizer code.
ld64_cmd = config.ld64_executable
asan_rtlib = get_asan_rtlib()
if asan_rtlib:
  ld64_cmd = "DYLD_INSERT_LIBRARIES={} {}".format(asan_rtlib, ld64_cmd)
config.substitutions.append( ('%ld64', ld64_cmd) )

# OCaml substitutions.
# Support tests for both native and bytecode builds.
config.substitutions.append( ('%ocamlc',
    "%s ocamlc -cclib -L%s %s" %
        (config.ocamlfind_executable, config.llvm_lib_dir, config.ocaml_flags)) )
if config.have_ocamlopt:
    config.substitutions.append( ('%ocamlopt',
        "%s ocamlopt -cclib -L%s -cclib -Wl,-rpath,%s %s" %
            (config.ocamlfind_executable, config.llvm_lib_dir, config.llvm_lib_dir, config.ocaml_flags)) )
else:
    config.substitutions.append( ('%ocamlopt', "true" ) )

# For each occurrence of an llvm tool name as its own word, replace it
# with the full path to the build directory holding that tool.  This
# ensures that we are testing the tools just built and not some random
# tools that might happen to be in the user's PATH.  Thus this list
# includes every tool placed in $(LLVM_OBJ_ROOT)/$(BuildMode)/bin
# (llvm_tools_dir in lit parlance).

# Avoid matching RUN line fragments that are actually part of
# path names or options or whatever.
# The regex is a pre-assertion to avoid matching a preceding
# dot, hyphen, carat, or slash (.foo, -foo, etc.).  Some patterns
# also have a post-assertion to not match a trailing hyphen (foo-).
NOJUNK = r"(?<!\.|-|\^|/|<)"


def find_tool_substitution(pattern):
    # Extract the tool name from the pattern.  This relies on the tool
    # name being surrounded by \b word match operators.  If the
    # pattern starts with "| ", include it in the string to be
    # substituted.
    tool_match = re.match(r"^(\\)?((\| )?)\W+b([0-9A-Za-z-_]+)\\b\W*$",
                          pattern)
    tool_pipe = tool_match.group(2)
    tool_name = tool_match.group(4)
    # Did the user specify the tool path + arguments? This allows things like
    # llvm-lit "-Dllc=llc -enable-misched -verify-machineinstrs"
    tool_path = lit_config.params.get(tool_name)
    if tool_path is None:
        tool_path = lit.util.which(tool_name, config.llvm_tools_dir)
        if tool_path is None:
            return tool_name, tool_path, tool_pipe
    if (tool_name == "llc" and
       'LLVM_ENABLE_MACHINE_VERIFIER' in os.environ and
       os.environ['LLVM_ENABLE_MACHINE_VERIFIER'] == "1"):
        tool_path += " -verify-machineinstrs"
    if (tool_name == "llvm-go"):
        tool_path += " go=" + config.go_executable
    return tool_name, tool_path, tool_pipe


for pattern in [r"\bbugpoint\b(?!-)",
                NOJUNK + r"\bllc\b",
                r"\blli\b",
                r"\bllvm-ar\b",
                r"\bllvm-as\b",
                r"\bllvm-bcanalyzer\b",
                r"\bllvm-config\b",
                r"\bllvm-cov\b",
                r"\bllvm-cxxdump\b",
                r"\bllvm-cvtres\b",
                r"\bllvm-diff\b",
                r"\bllvm-dis\b",
                r"\bllvm-dsymutil\b",
                r"\bllvm-dwarfdump\b",
                r"\bllvm-extract\b",
                r"\bllvm-isel-fuzzer\b",
                r"\bllvm-lib\b",
                r"\bllvm-link\b",
                r"\bllvm-lto\b",
                r"\bllvm-lto2\b",
                r"\bllvm-mc\b",
                r"\bllvm-mcmarkup\b",
                r"\bllvm-modextract\b",
                r"\bllvm-nm\b",
                r"\bllvm-objcopy\b",
                r"\bllvm-objdump\b",
                r"\bllvm-pdbutil\b",
                r"\bllvm-profdata\b",
                r"\bllvm-ranlib\b",
                r"\bllvm-readobj\b",
                r"\bllvm-rtdyld\b",
                r"\bllvm-size\b",
                r"\bllvm-split\b",
                r"\bllvm-strings\b",
                r"\bllvm-tblgen\b",
                r"\bllvm-c-test\b",
                r"\bllvm-cxxfilt\b",
                r"\bllvm-xray\b",
                NOJUNK + r"\bllvm-symbolizer\b",
                NOJUNK + r"\bopt\b",
                r"\bFileCheck\b",
                r"\bobj2yaml\b",
                NOJUNK + r"\bsancov\b",
                NOJUNK + r"\bsanstats\b",
                r"\byaml2obj\b",
                r"\byaml-bench\b",
                r"\bverify-uselistorder\b",
                # Handle these specially as they are strings searched
                # for during testing.
                r"\| \bcount\b",
                r"\| \bnot\b"]:
    tool_name, tool_path, tool_pipe = find_tool_substitution(pattern)
    if not tool_path:
        # Warn, but still provide a substitution.
        lit_config.note('Did not find ' + tool_name + ' in ' + config.llvm_tools_dir)
        tool_path = config.llvm_tools_dir + '/' + tool_name
    config.substitutions.append((pattern, tool_pipe + tool_path))

# For tools that are optional depending on the config, we won't warn
# if they're missing.
for pattern in [r"\bllvm-go\b",
                r"\bllvm-mt\b",
                r"\bKaleidoscope-Ch3\b",
                r"\bKaleidoscope-Ch4\b",
                r"\bKaleidoscope-Ch5\b",
                r"\bKaleidoscope-Ch6\b",
                r"\bKaleidoscope-Ch7\b",
                r"\bKaleidoscope-Ch8\b"]:
    tool_name, tool_path, tool_pipe = find_tool_substitution(pattern)
    if not tool_path:
        # Provide a substitution anyway, for the sake of consistent errors.
        tool_path = config.llvm_tools_dir + '/' + tool_name
    config.substitutions.append((pattern, tool_pipe + tool_path))


### Targets

config.targets = frozenset(config.targets_to_build.split())

for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + '-registered-target')

### Features

# Others/can-execute.txt
if sys.platform not in ['win32']:
    config.available_features.add('can-execute')
    config.available_features.add('not_COFF')

# Loadable module
# FIXME: This should be supplied by Makefile or autoconf.
if sys.platform in ['win32', 'cygwin']:
    loadable_module = (config.enable_shared == 1)
else:
    loadable_module = True

if loadable_module:
    config.available_features.add('loadable_module')

# Static libraries are not built if BUILD_SHARED_LIBS is ON.
if not config.build_shared_libs:
    config.available_features.add("static-libs")

# Direct object generation
if not 'hexagon' in config.target_triple:
    config.available_features.add("object-emission")

# LLVM can be configured with an empty default triple
# Some tests are "generic" and require a valid default triple
if config.target_triple:
    config.available_features.add("default_triple")

import subprocess

def have_ld_plugin_support():
    if not os.path.exists(os.path.join(config.llvm_shlib_dir, 'LLVMgold.so')):
        return False

    ld_cmd = subprocess.Popen([config.gold_executable, '--help'], stdout = subprocess.PIPE, env={'LANG': 'C'})
    ld_out = ld_cmd.stdout.read().decode()
    ld_cmd.wait()

    if not '-plugin' in ld_out:
        return False

    # check that the used emulations are supported.
    emu_line = [l for l in ld_out.split('\n') if 'supported emulations' in l]
    if len(emu_line) != 1:
        return False
    emu_line = emu_line[0]
    fields = emu_line.split(':')
    if len(fields) != 3:
        return False
    emulations = fields[2].split()
    if 'elf_x86_64' not in emulations:
        return False
    if 'elf32ppc' in emulations:
        config.available_features.add('ld_emu_elf32ppc')

    ld_version = subprocess.Popen([config.gold_executable, '--version'], stdout = subprocess.PIPE, env={'LANG': 'C'})
    if not 'GNU gold' in ld_version.stdout.read().decode():
        return False
    ld_version.wait()

    return True

if have_ld_plugin_support():
    config.available_features.add('ld_plugin')

def have_ld64_plugin_support():
    if not config.llvm_tool_lto_build or config.ld64_executable == '':
        return False

    ld_cmd = subprocess.Popen([config.ld64_executable, '-v'], stderr = subprocess.PIPE)
    ld_out = ld_cmd.stderr.read().decode()
    ld_cmd.wait()

    if 'ld64' not in ld_out or 'LTO' not in ld_out:
        return False

    return True

if have_ld64_plugin_support():
    config.available_features.add('ld64_plugin')

# Ask llvm-config about asserts and global-isel.
llvm_config.feature_config(
  [('--assertion-mode', {'ON' : 'asserts'}),
   ('--has-global-isel', {'ON' : 'global-isel'})])

if 'darwin' == sys.platform:
    try:
        sysctl_cmd = subprocess.Popen(['sysctl', 'hw.optional.fma'],
                                    stdout = subprocess.PIPE)
    except OSError:
        print("Could not exec sysctl")
    result = sysctl_cmd.stdout.read().decode('ascii')
    if -1 != result.find("hw.optional.fma: 1"):
        config.available_features.add('fma3')
    sysctl_cmd.wait()

# .debug_frame is not emitted for targeting Windows x64.
if not re.match(r'^x86_64.*-(mingw32|windows-gnu|win32)', config.target_triple):
    config.available_features.add('debug_frame')

if config.have_libxar:
    config.available_features.add('xar')

if config.llvm_libxml2_enabled == "1":
    config.available_features.add('libxml2')
