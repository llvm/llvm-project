// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Argument attributes

// expected-error@below {{"llvm.sret" attribute attached to LLVM pointer argument of different type}}
llvm.func @invalid_sret_attr_type(%0 : !llvm.ptr<f32> {llvm.sret = !llvm.struct<(i32)>})
