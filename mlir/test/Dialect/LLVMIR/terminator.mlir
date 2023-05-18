// RUN: mlir-opt -pass-pipeline='builtin.module(func.func(canonicalize))' %s | FileCheck %s
// verify that terminators survive the canonicalizer

// CHECK-LABEL: @return
// CHECK: llvm.return
func.func @return() {
  llvm.return
}

// CHECK-LABEL: @control_flow
// CHECK: llvm.br
// CHECK: llvm.cond_br
// CHECK: llvm.return
func.func @control_flow(%cond : i1) {
  llvm.br ^bb1
^bb1:
  llvm.cond_br %cond, ^bb2, ^bb1
^bb2:
   llvm.return
}

