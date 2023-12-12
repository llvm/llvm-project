// RUN: mlir-opt %s -split-input-file -test-constant-fold -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: func @fold_and_merge
func.func @fold_and_merge() -> (i32, i32) {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 5 : i32

  // CHECK-NEXT: [[C:%.+]] = arith.constant 6 : i32 loc(#[[FusedLoc:.*]])
  %2 = arith.addi %0, %1 : i32 loc("fold_and_merge":0:0)

  %3 = arith.constant 6 : i32 loc("fold_and_merge":1:0)

  return %2, %3: i32, i32
// CHECK: } loc(#[[LocFunc:.*]])
} loc("fold_and_merge":2:0)

// CHECK-DAG: #[[LocConst0:.*]] = loc("fold_and_merge":0:0)
// CHECK-DAG: #[[LocConst1:.*]] = loc("fold_and_merge":1:0)
// CHECK-DAG: #[[LocFunc]] = loc("fold_and_merge":2:0)
// CHECK: #[[FusedLoc]] = loc(fused<"CSE">[#[[LocConst1]], #[[LocFunc]], #[[LocConst0]]])

// -----

// CHECK-LABEL: func @materialize_different_dialect
func.func @materialize_different_dialect() -> (f32, f32) {
  // CHECK: arith.constant 1.{{0*}}e+00 : f32 loc(#[[FusedLoc:.*]])
  %0 = arith.constant -1.0 : f32
  %1 = math.absf %0 : f32 loc("materialize_different_dialect":0:0)
  %2 = arith.constant 1.0 : f32 loc("materialize_different_dialect":1:0)

  return %1, %2: f32, f32
// CHECK: } loc(#[[LocFunc:.*]])
} loc("materialize_different_dialect":2:0)

// CHECK-DAG: #[[LocConst0:.*]] = loc("materialize_different_dialect":0:0)
// CHECK-DAG: #[[LocConst1:.*]] = loc("materialize_different_dialect":1:0)
// CHECK-DAG: #[[LocFunc]] = loc("materialize_different_dialect":2:0)
// CHECK: #[[FusedLoc]] = loc(fused<"CSE">[#[[LocConst1]], #[[LocFunc]], #[[LocConst0]]])

// -----

// CHECK-LABEL: func @materialize_in_front
func.func @materialize_in_front(%arg0: memref<8xi32>) {
  // CHECK-NEXT: arith.constant 6 : i32 loc(#[[FusedLoc:.*]])
  affine.for %arg1 = 0 to 8 {
    %1 = arith.constant 1 : i32
    %2 = arith.constant 5 : i32
    %3 = arith.addi %1, %2 : i32 loc("materialize_in_front":0:0)
    memref.store %3, %arg0[%arg1] : memref<8xi32>
  }
  // CHECK: return
  return
// CHECK-NEXT: } loc(#[[LocFunc:.*]])
} loc("materialize_in_front":1:0)

// CHECK-DAG: #[[LocConst0:.*]] = loc("materialize_in_front":0:0)
// CHECK-DAG: #[[LocFunc]] = loc("materialize_in_front":1:0)
// CHECK: #[[FusedLoc]] = loc(fused<"CSE">[#[[LocConst0]], #[[LocFunc]]])
