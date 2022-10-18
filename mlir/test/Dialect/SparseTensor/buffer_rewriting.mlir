// RUN: mlir-opt %s -split-input-file --sparse-buffer-rewrite  --canonicalize --cse | FileCheck %s

// CHECK-LABEL: func @sparse_push_back(
//  CHECK-SAME: %[[A:.*]]: memref<?xindex>,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> memref<?xf64> {
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[S:.*]] = memref.dim %[[B]], %[[C0]]
//       CHECK: %[[P:.*]] = memref.load %[[A]]{{\[}}%[[C2]]]
//       CHECK: %[[T:.*]] = arith.cmpi uge, %[[P]], %[[S]]
//       CHECK: %[[M:.*]] = scf.if %[[T]] -> (memref<?xf64>) {
//       CHECK:  %[[P1:.*]] = arith.muli %[[S]], %[[C2]]
//       CHECK:  %[[M2:.*]] = memref.realloc %[[B]](%[[P1]])
//       CHECK:  scf.yield %[[M2]] : memref<?xf64>
//       CHECK: } else {
//       CHECK:  scf.yield %[[B]] : memref<?xf64>
//       CHECK: }
//       CHECK: memref.store %[[C]], %[[M]]{{\[}}%[[P]]]
//       CHECK: %[[P2:.*]] = arith.addi %[[P]], %[[C1]]
//       CHECK: memref.store %[[P2]], %[[A]]{{\[}}%[[C2]]]
//       CHECK: return %[[M]] : memref<?xf64>
func.func @sparse_push_back(%arg0: memref<?xindex>, %arg1: memref<?xf64>, %arg2: f64) -> memref<?xf64> {
  %0 = sparse_tensor.push_back %arg0, %arg1, %arg2 {idx = 2 : index} : memref<?xindex>, memref<?xf64>, f64
  return %0 : memref<?xf64>
}

// -----

// CHECK-LABEL: func @sparse_push_back_inbound(
//  CHECK-SAME: %[[A:.*]]: memref<?xindex>,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> memref<?xf64> {
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//       CHECK: %[[P:.*]] = memref.load %[[A]]{{\[}}%[[C2]]]
//       CHECK: memref.store %[[C]], %[[B]]{{\[}}%[[P]]]
//       CHECK: %[[P2:.*]] = arith.addi %[[P]], %[[C1]]
//       CHECK: memref.store %[[P2]], %[[A]]{{\[}}%[[C2]]]
//       CHECK: return %[[B]] : memref<?xf64>
func.func @sparse_push_back_inbound(%arg0: memref<?xindex>, %arg1: memref<?xf64>, %arg2: f64) -> memref<?xf64> {
  %0 = sparse_tensor.push_back inbounds %arg0, %arg1, %arg2 {idx = 2 : index} : memref<?xindex>, memref<?xf64>, f64
  return %0 : memref<?xf64>
}

// -----

// CHECK-LABEL:   func.func private @_sparse_less_than_1_i8(
// CHECK-SAME:                                              %[[I:arg0]]: index,
// CHECK-SAME:                                              %[[J:.*]]: index,
// CHECK-SAME:                                              %[[X0:.*]]: memref<?xi8>) -> i1 {
// CHECK:           %[[VI:.*]] = memref.load %[[X0]]{{\[}}%[[I]]]
// CHECK:           %[[VJ:.*]] = memref.load %[[X0]]{{\[}}%[[J]]]
// CHECK:           %[[C:.*]] = arith.cmpi ult, %[[VI]], %[[VJ]]
// CHECK:           return %[[C]]
// CHECK:         }

// CHECK-LABEL:   func.func private @_sparse_may_swap_1_i8_f32_index(
// CHECK-SAME:                                                       %[[I:arg0]]: index,
// CHECK-SAME:                                                       %[[J:.*]]: index,
// CHECK-SAME:                                                       %[[X0:.*]]: memref<?xi8>,
// CHECK-SAME:                                                       %[[Y0:.*]]: memref<?xf32>,
// CHECK-SAME:                                                       %[[Y1:.*]]: memref<?xindex>) {
// CHECK:           %[[C:.*]] = arith.cmpi ne, %[[I]], %[[J]]
// CHECK:           scf.if %[[C]] {
// CHECK:             %[[Vx0i:.*]] = memref.load %[[X0]]{{\[}}%[[I]]]
// CHECK:             %[[Vx0j:.*]] = memref.load %[[X0]]{{\[}}%[[J]]]
// CHECK:             memref.store %[[Vx0j]], %[[X0]]{{\[}}%[[I]]]
// CHECK:             memref.store %[[Vx0i]], %[[X0]]{{\[}}%[[J]]]
// CHECK:             %[[Vy0i:.*]] = memref.load %[[Y0]]{{\[}}%[[I]]]
// CHECK:             %[[Vy0j:.*]] = memref.load %[[Y0]]{{\[}}%[[J]]]
// CHECK:             memref.store %[[Vy0j]], %[[Y0]]{{\[}}%[[I]]]
// CHECK:             memref.store %[[Vy0i]], %[[Y0]]{{\[}}%[[J]]]
// CHECK:             %[[Vy1i:.*]] = memref.load %[[Y1]]{{\[}}%[[I]]]
// CHECK:             %[[Vy1j:.*]] = memref.load %[[Y1]]{{\[}}%[[J]]]
// CHECK:             memref.store %[[Vy1j]], %[[Y1]]{{\[}}%[[I]]]
// CHECK:             memref.store %[[Vy1i]], %[[Y1]]{{\[}}%[[J]]]
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_sparse_partition_1_i8_f32_index(
// CHECK-SAME:                                                        %[[L:arg0]]: index,
// CHECK-SAME:                                                        %[[H:.*]]: index,
// CHECK-SAME:                                                        %[[X0:.*]]: memref<?xi8>,
// CHECK-SAME:                                                        %[[Y0:.*]]: memref<?xf32>,
// CHECK-SAME:                                                        %[[Y1:.*]]: memref<?xindex>) -> index {
// CHECK:           %[[C1:.*]] = arith.constant 1
// CHECK:           %[[I:.*]] = arith.subi %[[L]], %[[C1]]
// CHECK:           %[[Hm1:.*]] = arith.subi %[[H]], %[[C1]]
// CHECK:           %[[I3:.*]] = scf.for %[[J:.*]] = %[[L]] to %[[Hm1]] step %[[C1]] iter_args(%[[I2:.*]] = %[[I]]) -> (index) {
// CHECK:             %[[COND:.*]] = func.call @_sparse_less_than_1_i8(%[[J]], %[[Hm1]], %[[X0]])
// CHECK:             %[[IF:.*]] = scf.if %[[COND]] -> (index) {
// CHECK:               %[[Ip1:.*]] = arith.addi %[[I2]], %[[C1]]
// CHECK:               func.call @_sparse_may_swap_1_i8_f32_index(%[[Ip1]], %[[J]], %[[X0]], %[[Y0]], %[[Y1]])
// CHECK:               scf.yield %[[Ip1]]
// CHECK:             } else {
// CHECK:               scf.yield %[[I2]]
// CHECK:             }
// CHECK:             scf.yield %[[IF:.*]]
// CHECK:           }
// CHECK:           %[[I3p1:.*]] = arith.addi %[[I3:.*]], %[[C1]] : index
// CHECK:           call @_sparse_may_swap_1_i8_f32_index(%[[I3p1]], %[[Hm1]], %[[X0]], %[[Y0]], %[[Y1]])
// CHECK:           return %[[I3p1]]
// CHECK:         }

// CHECK-LABEL:   func.func private @_sparse_sort_nonstable_1_i8_f32_index(
// CHECK-SAME:                                                   %[[L:arg0]]: index,
// CHECK-SAME:                                                   %[[H:.*]]: index,
// CHECK-SAME:                                                   %[[X0:.*]]: memref<?xi8>,
// CHECK-SAME:                                                   %[[Y0:.*]]: memref<?xf32>,
// CHECK-SAME:                                                   %[[Y1:.*]]: memref<?xindex>) {
// CHECK:           %[[C1:.*]] = arith.constant 1
// CHECK:           %[[COND:.*]] = arith.cmpi ult, %[[L]], %[[H]]
// CHECK:           scf.if %[[COND]] {
// CHECK:             %[[P:.*]] = func.call @_sparse_partition_1_i8_f32_index(%[[L]], %[[H]], %[[X0]], %[[Y0]], %[[Y1]])
// CHECK:             func.call @_sparse_sort_nonstable_1_i8_f32_index(%[[L]], %[[P]], %[[X0]], %[[Y0]], %[[Y1]])
// CHECK:             %[[P2:.*]] = arith.addi %[[P]], %[[C1]] : index
// CHECK:             func.call @_sparse_sort_nonstable_1_i8_f32_index(%[[P2]], %[[H]], %[[X0]], %[[Y0]], %[[Y1]])
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @sparse_sort_1d2v(
// CHECK-SAME:                                %[[N:.*]]: index,
// CHECK-SAME:                                %[[X0:.*]]: memref<10xi8>,
// CHECK-SAME:                                %[[Y0:.*]]: memref<?xf32>,
// CHECK-SAME:                                %[[Y1:.*]]: memref<10xindex>) -> (memref<10xi8>, memref<?xf32>, memref<10xindex>) {
// CHECK:           %[[C0:.*]] = arith.constant 0
// CHECK:           %[[DX0:.*]] = memref.cast %[[X0]] : memref<10xi8> to memref<?xi8>
// CHECK:           %[[DY1:.*]] = memref.cast %[[Y1]] : memref<10xindex> to memref<?xindex>
// CHECK:           call @_sparse_sort_nonstable_1_i8_f32_index(%[[C0]], %[[N]], %[[DX0]], %[[Y0]], %[[DY1]])
// CHECK:           return %[[X0]], %[[Y0]], %[[Y1]]
// CHECK:         }
func.func @sparse_sort_1d2v(%arg0: index, %arg1: memref<10xi8>, %arg2: memref<?xf32>, %arg3: memref<10xindex>)
   -> (memref<10xi8>, memref<?xf32>, memref<10xindex>) {
  sparse_tensor.sort %arg0, %arg1 jointly %arg2, %arg3 : memref<10xi8> jointly memref<?xf32>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xi8>, memref<?xf32>, memref<10xindex>
}

// -----

// Only check the generated supporting function now. We have integration test
// to verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_less_than_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> i1 {
// CHECK-DAG:     func.func private @_sparse_may_swap_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-DAG:     func.func private @_sparse_partition_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_nonstable_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-LABEL:   func.func @sparse_sort_3d
func.func @sparse_sort_3d(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_less_than_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> i1 {
// CHECK-DAG:     func.func private @_sparse_binary_search_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-LABEL:   func.func @sparse_sort_3d_stable
func.func @sparse_sort_3d_stable(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort stable %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}
