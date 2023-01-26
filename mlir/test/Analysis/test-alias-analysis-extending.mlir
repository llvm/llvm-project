// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-alias-analysis-extending))' -split-input-file -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "restrict"
// CHECK-DAG: func.region0#0 <-> func.region0#1: NoAlias

// CHECK-DAG: view1#0 <-> view2#0: NoAlias
// CHECK-DAG: view1#0 <-> func.region0#0: MustAlias
// CHECK-DAG: view1#0 <-> func.region0#1: NoAlias
// CHECK-DAG: view2#0 <-> func.region0#0: NoAlias
// CHECK-DAG: view2#0 <-> func.region0#1: MustAlias
func.func @restrict(%arg: memref<?xf32>, %arg1: memref<?xf32> {local_alias_analysis.restrict}) attributes {test.ptr = "func"} {
  %0 = memref.subview %arg[0][2][1] {test.ptr = "view1"} : memref<?xf32> to memref<2xf32>
  %1 = memref.subview %arg1[0][2][1] {test.ptr = "view2"} : memref<?xf32> to memref<2xf32>
  return
}
