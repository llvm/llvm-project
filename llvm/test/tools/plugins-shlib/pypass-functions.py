# REQUIRES: native, system-linux, llvm-dylib, plugins, pypass-plugin
#
# We can run on inidivdual functions instead of whole modules
# RUN: opt -load-pass-plugin=%plugindir/pypass-plugin%pluginext \
# RUN:     -pypass-script=%s -pypass-dylib=%libpython \
# RUN:     -passes=pypass -disable-output %S/Inputs/foobar.ll | FileCheck %s

# CHECK: 0x{{[0-9a-f]+}} Function
# CHECK: 0x{{[0-9a-f]+}} Function


def registerModulePipelineParsingCallback():
    """Call run for each module that goes through the pipeline"""
    return False


def registerFunctionPipelineParsingCallback():
    """Call run for each function that goes through the pipeline"""
    return True


def run(input, ctx, stage):
    print(f"0x{input:016x} {stage}")
