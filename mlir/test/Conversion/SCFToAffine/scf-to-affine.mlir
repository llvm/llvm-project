// RUN: mlir-opt -raise-scf-to-affine -split-input-file %s | FileCheck %s

// CHECK: #[[$UB_MAP:.+]] = affine_map<(d0, d1)[s0] -> ((d1 - d0 + s0 - 1) floordiv s0)>
// CHECK: #[[$IV_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 * s0)>
// CHECK-LABEL: @generic_loop
// CHECK-SAME: %[[ARR:.*]]: memref<?xi32>, %[[LOWER:.*]]: index, %[[UPPER:.*]]: index, %[[STEP:.*]]: index
func.func @generic_loop(%arr: memref<?xi32>, %lower: index, %upper: index, %step: index) {
// CHECK: affine.for %[[IV:.*]] = 0 to #[[$UB_MAP]](%[[LOWER]], %[[UPPER]])[%[[STEP]]] {
// CHECK:   %[[IDX:.*]] = affine.apply #[[$IV_MAP]](%[[LOWER]], %[[IV]])[%[[STEP]]]
// CHECK:   memref.store %{{.*}}, %[[ARR]][%[[IDX]]] : memref<?xi32>
// CHECK: }
  %c0_i32 = arith.constant 0 : i32
  scf.for %idx = %lower to %upper step %step {
    memref.store %c0_i32, %arr[%idx] : memref<?xi32>
  }
  return
}

// -----

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @loop_with_constant_step
// CHECK-SAME: %[[ARR:.*]]: memref<?xi32>, %[[LOWER:.*]]: index, %[[UPPER:.*]]: index
func.func @loop_with_constant_step(%arr: memref<?xi32>, %lower: index, %upper: index) {
// CHECK: affine.for %[[IDX:.*]] = #[[$MAP]](%[[LOWER]]) to #[[$MAP]](%[[UPPER]]) step 3 {
// CHECK:   memref.store %{{.*}}, %[[ARR]][%[[IDX]]] : memref<?xi32>
// CHECK: }
  %c0_i32 = arith.constant 0 : i32
  %c3 = arith.constant 3 : index
  scf.for %idx = %lower to %upper step %c3 {
    memref.store %c0_i32, %arr[%idx] : memref<?xi32>
  }
  return
}

// -----

// CHECK-LABEL: @nested_loop
func.func @nested_loop(%arg0: memref<?x?xi32>, %upper1: index, %upper2: index) {
// CHECK: affine.for
// CHECK:   affine.for
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %upper1 step %c1 {
    scf.for %j = %c0 to %upper2 step %c1 {
      memref.store %c0_i32, %arg0[%i, %j] : memref<?x?xi32>
    }
  }
  return
}
