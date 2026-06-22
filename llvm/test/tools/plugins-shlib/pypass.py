# REQUIRES: native, system-linux, llvm-dylib, plugins, pypass-plugin

# We can use environment variables for parameters
# RUN: env LLVM_PYPASS_DYLIB=%libpython \
# RUN: env LLVM_PYPASS_SCRIPT=%s \
# RUN:   opt -load-pass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:       -passes=pypass -disable-output %S/Inputs/foobar.ll | FileCheck %s
#
# We can use command-line arguments for parameters
# RUN:   opt -load-pass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:       -pypass-script=%s -pypass-dylib=%libpython \
# RUN:       -passes=pypass -disable-output %S/Inputs/foobar.ll | FileCheck %s
#
# Loading the plugin twice causes no issues
# RUN:   opt -load-pass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:       -load-pass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:       -pypass-script=%s -pypass-dylib=%libpython \
# RUN:       -passes=pypass -disable-output %S/Inputs/foobar.ll | FileCheck %s
#
# CHECK: Python version: 3
# CHECK: 0x{{[0-9a-f]+}} Module

# Fail gracefully for invalid libpython path
# RUN: not opt -load-pass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:         -pypass-script=%s -pypass-dylib=invalid \
# RUN:         -passes=pypass -disable-output %S/Inputs/foobar.ll 2>&1 | FileCheck --check-prefix=INVALID-DYLIB %s
#
# INVALID-DYLIB: Failed to load Python shared library: 'invalid'

# Fail gracefully for invalid script path
# RUN: not opt -load-pass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:         -pypass-script=invalid -pypass-dylib=%libpython \
# RUN:         -passes=pypass -disable-output %S/Inputs/foobar.ll 2>&1 | FileCheck --check-prefix=INVALID-SCRIPT %s
#
# INVALID-SCRIPT: Failed to locate script file: 'invalid'

# We can import modules relative to the script directory
from Inputs.mymod import pyversion

pyversion()


# We don't forward the actual pipeline parsing callback (yet)
def registerModulePipelineParsingCallback():
    """Call run for each module that goes through the pipeline"""
    return True


# We get the addresses of the C API LLVMModule and LLVMContext
def run(mod, ctx, stage):
    print(f"0x{mod:016x} {stage}")
