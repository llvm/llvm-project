// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: func @merge_constants
func.func @merge_constants() -> (index, index, index, index, index, index, index) {
  // CHECK-NEXT: arith.constant 42 : index loc(#[[FusedLoc:.*]])
  %0 = arith.constant 42 : index loc("merge_constants":0:0)
  %1 = arith.constant 42 : index loc("merge_constants":1:0)
  %2 = arith.constant 42 : index loc("merge_constants":2:0)
  %3 = arith.constant 42 : index loc("merge_constants":2:0) // repeated loc
  %4 = arith.constant 43 : index loc(fused<"some_label">["merge_constants":3:0])
  %5 = arith.constant 43 : index loc(fused<"some_label">["merge_constants":3:0])
  %6 = arith.constant 43 : index loc(fused<"some_other_label">["merge_constants":3:0])
  return %0, %1, %2, %3, %4, %5, %6 : index, index, index, index, index, index, index
}

// CHECK-DAG: #[[LocConst0:.*]] = loc("merge_constants":0:0)
// CHECK-DAG: #[[LocConst1:.*]] = loc("merge_constants":1:0)
// CHECK-DAG: #[[LocConst2:.*]] = loc("merge_constants":2:0)
// CHECK-DAG: #[[LocConst3:.*]] = loc("merge_constants":3:0)
// CHECK-DAG: #[[FusedLoc_CSE_1:.*]] = loc(fused<"CSE">[#[[LocConst0]], #[[LocConst1]], #[[LocConst2]]])
// CHECK-DAG: #[[FusedLoc_Some_Label:.*]] = loc(fused<"some_label">[#[[LocConst3]]])
// CHECK-DAG: #[[FusedLoc_Some_Other_Label:.*]] = loc(fused<"some_other_label">[#[[LocConst3]]])
// CHECK-DAG: #[[FusedLoc_CSE_2:.*]] = loc(fused<"CSE">[#[[FusedLoc_Some_Label]], #[[FusedLoc_Some_Other_Label]]])

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
