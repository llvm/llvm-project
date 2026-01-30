// RUN: mlir-opt %s -split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @fsub_canonicalize
llvm.func @fsub_canonicalize(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: %[[NEG_ARG1:.*]] = llvm.fneg %arg1 : f32
  // CHECK: %[[ADD_RESULT:.*]] = nvvm.fadd %arg0, %[[NEG_ARG1]] : f32, f32 -> f32
  %0 = nvvm.fsub %arg0, %arg1 : f32, f32 -> f32
  llvm.return %0 : f32
}
