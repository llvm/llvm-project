// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: func @merge_constants
func.func @merge_constants() -> (index, index, index, index) {
  // CHECK-NEXT: arith.constant 42 : index loc(#[[UnknownLoc:.*]])
  %0 = arith.constant 42 : index loc("merge_constants":0:0)
  %1 = arith.constant 42 : index loc("merge_constants":1:0)
  %2 = arith.constant 42 : index loc("merge_constants":2:0)
  %3 = arith.constant 42 : index loc("merge_constants":2:0)
  return %0, %1, %2, %3 : index, index, index, index
}
// CHECK: #[[UnknownLoc]] = loc(unknown)

// -----

// CHECK-LABEL: func @simple_hoist
func.func @simple_hoist(%arg0: memref<8xi32>) -> i32 {
  // CHECK: arith.constant 88 : i32 loc(#[[UnknownLoc:.*]])
  // CHECK: arith.constant 42 : i32 loc(#[[ConstLoc0:.*]])
  // CHECK: arith.constant 0 : index loc(#[[ConstLoc1:.*]])
  %0 = arith.constant 42 : i32 loc("simple_hoist":0:0)
  %1 = arith.constant 0 : index loc("simple_hoist":1:0)
  memref.store %0, %arg0[%1] : memref<8xi32>

  %2 = arith.constant 88 : i32 loc("simple_hoist":2:0)

  return %2 : i32
}
// CHECK-DAG: #[[UnknownLoc]] = loc(unknown)
// CHECK-DAG: #[[ConstLoc0]] = loc("simple_hoist":0:0)
// CHECK-DAG: #[[ConstLoc1]] = loc("simple_hoist":1:0)

// -----

// CHECK-LABEL: func @hoist_and_merge
func.func @hoist_and_merge(%arg0: memref<8xi32>) {
  // CHECK-NEXT: arith.constant 42 : i32 loc(#[[UnknownLoc:.*]])
  affine.for %arg1 = 0 to 8 {
    %0 = arith.constant 42 : i32 loc("hoist_and_merge":0:0)
    %1 = arith.constant 42 : i32 loc("hoist_and_merge":1:0)
    memref.store %0, %arg0[%arg1] : memref<8xi32>
    memref.store %1, %arg0[%arg1] : memref<8xi32>
  }
  return
} loc("hoist_and_merge":2:0)
// CHECK: #[[UnknownLoc]] = loc(unknown)
