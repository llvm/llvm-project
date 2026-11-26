// RUN: mlir-opt %s -split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @subf_canonicalize
llvm.func @subf_canonicalize(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: %[[NEG_ARG1:.*]] = llvm.fneg %arg1 : f32
  // CHECK: %[[ADD_RESULT:.*]] = nvvm.addf %arg0, %[[NEG_ARG1]] : f32
  %0 = nvvm.subf %arg0, %arg1 : f32
  llvm.return %0 : f32
}

// -----

// CHECK-LABEL: @mulf_commutative
llvm.func @mulf_commutative(%arg0 : f32) -> f32 {
  // CHECK: %[[CST:.*]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
  // CHECK: nvvm.mulf %arg0, %[[CST]] rnd = <rz> : f32
  %cst = llvm.mlir.constant(2.0 : f32) : f32
  %0 = nvvm.mulf %cst, %arg0 rnd = <rz> : f32
  llvm.return %0 : f32
}
