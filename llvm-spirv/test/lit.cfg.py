# -*- Python -*-

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'LLVM_SPIRV'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.cl', '.ll', '.spt']

# excludes: A list of directories  and fles to exclude from the testsuite.
config.excludes = ['CMakeLists.txt']

if not config.spirv_skip_debug_info_tests:
    # Direct object generation.
    config.available_features.add('object-emission')
    
    # LLVM can be configured with an empty default triple.
    # Some tests are "generic" and require a valid default triple.
    if config.target_triple:
        config.available_features.add('default_triple')
    
    # Ask llvm-config about asserts.
    llvm_config.feature_config([('--assertion-mode', {'ON': 'asserts'})])

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.test_run_dir, 'test_output')

llvm_config.use_default_substitutions()

llvm_config.use_clang()

config.substitutions.append(('%PATH%', config.environment['PATH']))

tool_dirs = [config.llvm_tools_dir, config.llvm_spirv_dir]

tools = ['llvm-as', 'llvm-dis', 'llvm-spirv', 'not']
if not config.spirv_skip_debug_info_tests:
    tools.extend(['llc', 'llvm-dwarfdump', 'llvm-objdump', 'llvm-readelf', 'llvm-readobj'])

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.spirv_tools_have_spirv_val:
    new_ld_library_path = os.path.pathsep.join((config.spirv_tools_lib_dir, config.environment['LD_LIBRARY_PATH']))
    config.environment['LD_LIBRARY_PATH'] = new_ld_library_path
    llvm_config.add_tool_substitutions(['spirv-val'], [config.spirv_tools_bin_dir])
else:
    config.substitutions.append(('spirv-val', ':'))
