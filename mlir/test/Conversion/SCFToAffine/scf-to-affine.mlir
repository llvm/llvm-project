// RUN: mlir-opt -raise-scf-to-affine -split-input-file %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1)[s0] -> ((d1 - d0 + s0 - 1) floordiv s0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 * s0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @simple_loop(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xi32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<3xindex>) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = memref.load %[[ARG1]]{{\[}}%[[VAL_1]]] : memref<3xindex>
// CHECK:           %[[VAL_5:.*]] = memref.load %[[ARG1]]{{\[}}%[[VAL_2]]] : memref<3xindex>
// CHECK:           %[[VAL_6:.*]] = memref.load %[[ARG1]]{{\[}}%[[VAL_3]]] : memref<3xindex>
// CHECK:           affine.for %[[VAL_7:.*]] = 0 to #[[$ATTR_0]](%[[VAL_4]], %[[VAL_5]]){{\[}}%[[VAL_6]]] {
// CHECK:             %[[VAL_8:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_4]], %[[VAL_7]]){{\[}}%[[VAL_6]]]
// CHECK:             memref.store %[[VAL_0]], %[[ARG0]]{{\[}}%[[VAL_8]]] : memref<?xi32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @simple_loop(%arg0: memref<?xi32>, %arg1: memref<3xindex>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.load %arg1[%c0] : memref<3xindex>
  %1 = memref.load %arg1[%c1] : memref<3xindex>
  %2 = memref.load %arg1[%c2] : memref<3xindex>
  scf.for %arg2 = %0 to %1 step %2 {
    memref.store %c0_i32, %arg0[%arg2] : memref<?xi32>
  }
  return
}

// CHECK-LABEL:   func.func @loop_with_constant_step(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xi32>,
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK:           affine.for %[[VAL_2:.*]] = #[[$ATTR_2]](%[[ARG1]]) to #[[$ATTR_2]](%[[ARG2]]) step 3 {
// CHECK:             memref.store %[[VAL_0]], %[[ARG0]]{{\[}}%[[VAL_2]]] : memref<?xi32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @loop_with_constant_step(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0_i32 = arith.constant 0 : i32
  %c3 = arith.constant 3 : index
  scf.for %arg3 = %arg1 to %arg2 step %c3 {
    memref.store %c0_i32, %arg0[%arg3] : memref<?xi32>
  }
  return
}

