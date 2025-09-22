// RUN: mlir-opt -convert-to-emitc %s -split-input-file -verify-diagnostics

func.func @block_args(%arg0: i1, %arg1: index, %arg2: index) -> index {
    // expected-error @+1 {{type mismatch for bb argument #0 of successor #0}}
    cf.cond_br %arg0, ^bb1(%arg1: index), ^bb2(%arg2: index)
^bb1(%0: index):
    return %0 : index
^bb2(%1: index):
    return %1 : index
}
