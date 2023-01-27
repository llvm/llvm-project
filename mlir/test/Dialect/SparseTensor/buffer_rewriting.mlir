// RUN: mlir-opt %s -split-input-file --sparse-buffer-rewrite  --canonicalize --cse | FileCheck %s

// CHECK-LABEL: func @sparse_push_back(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> (memref<?xf64>, index) {
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[P1:.*]] = memref.dim %[[B]], %[[C0]]
//       CHECK: %[[S2:.*]] = arith.addi %[[A]], %[[C1]] : index
//       CHECK: %[[T:.*]] = arith.cmpi ugt, %[[S2]], %[[P1]]
//       CHECK: %[[M:.*]] = scf.if %[[T]] -> (memref<?xf64>) {
//       CHECK:  %[[P2:.*]] = arith.muli %[[P1]], %[[C2]]
//       CHECK:  %[[M2:.*]] = memref.realloc %[[B]](%[[P2]])
//       CHECK:  scf.yield %[[M2]] : memref<?xf64>
//       CHECK: } else {
//       CHECK:  scf.yield %[[B]] : memref<?xf64>
//       CHECK: }
//       CHECK: memref.store %[[C]], %[[M]]{{\[}}%[[A]]]
//       CHECK: return %[[M]], %[[S2]]
func.func @sparse_push_back(%arg0: index, %arg1: memref<?xf64>, %arg2: f64) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back %arg0, %arg1, %arg2 : index, memref<?xf64>, f64
  return %0#0, %0#1 : memref<?xf64>, index
}

// -----

// CHECK-LABEL: func @sparse_push_back_n(
//  CHECK-SAME: %[[S1:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64,
//  CHECK-SAME: %[[D:.*]]: index) -> (memref<?xf64>, index) {
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//       CHECK: %[[P1:.*]] = memref.dim %[[B]], %[[C0]]
//       CHECK: %[[S2:.*]] = arith.addi %[[S1]], %[[D]] : index
//       CHECK: %[[T:.*]] = arith.cmpi ugt, %[[S2]], %[[P1]]
//       CHECK: %[[M:.*]] = scf.if %[[T]] -> (memref<?xf64>) {
//       CHECK:   %[[P2:.*]] = scf.while (%[[I:.*]] = %[[P1]]) : (index) -> index {
//       CHECK:     %[[P3:.*]] = arith.muli %[[I]], %[[C2]] : index
//       CHECK:     %[[T2:.*]] = arith.cmpi ugt, %[[S2]], %[[P3]] : index
//       CHECK:     scf.condition(%[[T2]]) %[[P3]] : index
//       CHECK:   } do {
//       CHECK:     ^bb0(%[[I2:.*]]: index):
//       CHECK:     scf.yield %[[I2]] : index
//       CHECK:   }
//       CHECK:  %[[M2:.*]] = memref.realloc %[[B]](%[[P2]])
//       CHECK:  scf.yield %[[M2]] : memref<?xf64>
//       CHECK: } else {
//       CHECK:  scf.yield %[[B]] : memref<?xf64>
//       CHECK: }
//       CHECK: %[[S:.*]] = memref.subview %[[M]]{{\[}}%[[S1]]] {{\[}}%[[D]]] [1]
//       CHECK: linalg.fill ins(%[[C]] : f64) outs(%[[S]]
//       CHECK: return %[[M]], %[[S2]] : memref<?xf64>, index
func.func @sparse_push_back_n(%arg0: index, %arg1: memref<?xf64>, %arg2: f64, %arg3: index) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back %arg0, %arg1, %arg2, %arg3 : index, memref<?xf64>, f64, index
  return %0#0, %0#1 : memref<?xf64>, index
}

// -----

// CHECK-LABEL: func @sparse_push_back_inbound(
//  CHECK-SAME: %[[S1:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> (memref<?xf64>, index) {
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: %[[S2:.*]] = arith.addi %[[S1]], %[[C1]]
//       CHECK: memref.store %[[C]], %[[B]]{{\[}}%[[S1]]]
//       CHECK: return %[[B]], %[[S2]] : memref<?xf64>, index
func.func @sparse_push_back_inbound(%arg0: index, %arg1: memref<?xf64>, %arg2: f64) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back inbounds %arg0, %arg1, %arg2 : index, memref<?xf64>, f64
  return %0#0, %0#1 : memref<?xf64>, index
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

// CHECK-LABEL:   func.func private @_sparse_compare_eq_1_i8(
// CHECK-SAME:                                               %[[I:arg0]]: index,
// CHECK-SAME:                                               %[[J:.*]]: index,
// CHECK-SAME:                                               %[[X0:.*]]: memref<?xi8>) -> i1 {
// CHECK:           %[[VI:.*]] = memref.load %[[X0]]{{\[}}%[[I]]]
// CHECK:           %[[VJ:.*]] = memref.load %[[X0]]{{\[}}%[[J]]]
// CHECK:           %[[C:.*]] = arith.cmpi eq, %[[VI]], %[[VJ]]
// CHECK:           return %[[C]]
// CHECK:         }

// CHECK-LABEL:   func.func private @_sparse_partition_1_i8_f32_index(
// CHECK-SAME:                                                        %[[L:arg0]]: index,
// CHECK-SAME:                                                        %[[H:.*]]: index,
// CHECK-SAME:                                                        %[[X0:.*]]: memref<?xi8>,
// CHECK-SAME:                                                        %[[Y0:.*]]: memref<?xf32>,
// CHECK-SAME:                                                        %[[Y1:.*]]: memref<?xindex>) -> index {
// CHECK:           %[[C1:.*]] = arith.constant 1
// CHECK:           %[[VAL_6:.*]] = arith.constant -
// CHECK:           %[[SUM:.*]] = arith.addi %[[L]], %[[H]]
// CHECK:           %[[P:.*]] = arith.shrui %[[SUM]], %[[C1]]
// CHECK:           %[[J:.*]] = arith.subi %[[H]], %[[C1]]
// CHECK:           %[[W:.*]]:3 = scf.while (%[[Ib:.*]] = %[[L]], %[[Jb:.*]] = %[[J]], %[[pb:.*]] = %[[P]]) : (index, index, index) -> (index, index, index) {
// CHECK:             %[[Cn:.*]] = arith.cmpi ult, %[[Ib]], %[[Jb]]
// CHECK:             scf.condition(%[[Cn]]) %[[Ib]], %[[Jb]], %[[pb]]
// CHECK:           } do {
// CHECK:           ^bb0(%[[Ia:.*]]: index, %[[Ja:.*]]: index, %[[Pa:.*]]: index):
// CHECK:             %[[I2:.*]] = scf.while
// CHECK:             %[[Ieq:.*]] = func.call @_sparse_compare_eq_1_i8(%[[I2:.*]], %[[Pa]], %[[X0]])
// CHECK:             %[[J2:.*]] = scf.while
// CHECK:             %[[Jeq:.*]] = func.call @_sparse_compare_eq_1_i8(%[[J2:.*]], %[[Pa]], %[[X0]])
// CHECK:             %[[Cn2:.*]] = arith.cmpi ult, %[[I2]], %[[J2]]
// CHECK:             %[[If:.*]]:3 = scf.if %[[Cn2]] -> (index, index, index) {
// CHECK:             } else {
// CHECK:               scf.yield %[[I2]], %[[J2]], %[[Pa]]
// CHECK:             }
// CHECK:             scf.yield %[[If:.*]]#0, %[[If]]#1, %[[If]]#2
// CHECK:           }
// CHECK:           return %[[W:.*]]#2
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
  sparse_tensor.sort hybrid_quick_sort %arg0, %arg1 jointly %arg2, %arg3 : memref<10xi8> jointly memref<?xf32>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xi8>, memref<?xf32>, memref<10xindex>
}

// -----

// Only check the generated supporting function now. We have integration test
// to verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_less_than_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> i1 {
// CHECK-DAG:     func.func private @_sparse_compare_eq_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> i1 {
// CHECK-DAG:     func.func private @_sparse_partition_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_nonstable_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-LABEL:   func.func @sparse_sort_3d
func.func @sparse_sort_3d(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort hybrid_quick_sort %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
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
  sparse_tensor.sort insertion_sort_stable %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_less_than_2_index_coo_1(%arg0: index, %arg1: index, %arg2: memref<?xindex>) -> i1 {
// CHECK-DAG:     func.func private @_sparse_compare_eq_2_index_coo_1(%arg0: index, %arg1: index, %arg2: memref<?xindex>) -> i1 {
// CHECK-DAG:     func.func private @_sparse_partition_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_nonstable_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-LABEL:   func.func @sparse_sort_coo
func.func @sparse_sort_coo(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo hybrid_quick_sort %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_less_than_2_index_coo_1(%arg0: index, %arg1: index, %arg2: memref<?xindex>) -> i1 {
// CHECK-DAG:     func.func private @_sparse_binary_search_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-LABEL:   func.func @sparse_sort_coo_stable
func.func @sparse_sort_coo_stable(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo insertion_sort_stable %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}

