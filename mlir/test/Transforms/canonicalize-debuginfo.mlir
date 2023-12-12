// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: func @merge_constants
func.func @merge_constants() -> (index, index, index, index) {
  // CHECK-NEXT: arith.constant 42 : index loc(#[[FusedLoc:.*]])
  %0 = arith.constant 42 : index loc("merge_constants":0:0)
  %1 = arith.constant 42 : index loc("merge_constants":1:0)
  %2 = arith.constant 42 : index loc("merge_constants":2:0)
  %3 = arith.constant 42 : index loc("merge_constants":2:0) // repeated loc
  return %0, %1, %2, %3: index, index, index, index
}

// CHECK-DAG: #[[LocConst0:.*]] = loc("merge_constants":0:0)
// CHECK-DAG: #[[LocConst1:.*]] = loc("merge_constants":1:0)
// CHECK-DAG: #[[LocConst2:.*]] = loc("merge_constants":2:0)
// CHECK: #[[FusedLoc]] = loc(fused<"CSE">[#[[LocConst0]], #[[LocConst1]], #[[LocConst2]]])

// -----

// CHECK-LABEL: func @hoist_constant
func.func @hoist_constant(%arg0: memref<8xi32>) {
  // CHECK-NEXT: arith.constant 42 : i32 loc(#[[FusedLoc:.*]])
  affine.for %arg1 = 0 to 8 {
    %0 = arith.constant 42 : i32 loc("hoist_constant":0:0)
    %1 = arith.constant 42 : i32 loc("hoist_constant":1:0)
    memref.store %0, %arg0[%arg1] : memref<8xi32>
    memref.store %1, %arg0[%arg1] : memref<8xi32>
  }
  return
}

// CHECK-DAG: #[[LocConst0:.*]] = loc("hoist_constant":0:0)
// CHECK-DAG: #[[LocConst1:.*]] = loc("hoist_constant":1:0)
// CHECK: #[[FusedLoc]] = loc(fused<"CSE">[#[[LocConst0]], #[[LocConst1]]])
