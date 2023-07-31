// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-ub-to-llvm))" %s -split-input-file | FileCheck %s

// CHECK-LABEL: @check_poison
func.func @check_poison() {
// CHECK: {{.*}} = llvm.mlir.poison : i64
  %0 = ub.poison : index
// CHECK: {{.*}} = llvm.mlir.poison : i16
  %1 = ub.poison : i16
// CHECK: {{.*}} = llvm.mlir.poison : f64
  %2 = ub.poison : f64
  return
}
