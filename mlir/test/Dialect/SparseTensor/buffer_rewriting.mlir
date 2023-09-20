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

// CHECK-LABEL:   func.func private @_sparse_partition_1_i8_f32_index
// CHECK-LABEL:   func.func private @_sparse_qsort_1_i8_f32_index
// CHECK-LABEL:   func.func @sparse_sort_1d2v_quick
func.func @sparse_sort_1d2v_quick(%arg0: index, %arg1: memref<10xi8>, %arg2: memref<?xf32>, %arg3: memref<10xindex>)
   -> (memref<10xi8>, memref<?xf32>, memref<10xindex>) {
  sparse_tensor.sort quick_sort %arg0, %arg1 jointly %arg2, %arg3 : memref<10xi8> jointly memref<?xf32>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xi8>, memref<?xf32>, memref<10xindex>
}

// -----

// Only check the generated supporting function now. We have integration test
// to verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_partition_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_qsort_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-LABEL:   func.func @sparse_sort_3d_quick
func.func @sparse_sort_3d_quick(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort quick_sort %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting function now. We have integration test
// to verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_binary_search_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-DAG:     func.func private @_sparse_shift_down_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: index) {
// CHECK-DAG:     func.func private @_sparse_heap_sort_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-DAG:     func.func private @_sparse_partition_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_hybrid_qsort_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: i64) {
// CHECK-LABEL:   func.func @sparse_sort_3d_hybrid
func.func @sparse_sort_3d_hybrid(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort hybrid_quick_sort %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
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
// CHECK-DAG:     func.func private @_sparse_shift_down_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: index) {
// CHECK-DAG:     func.func private @_sparse_heap_sort_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-LABEL:   func.func @sparse_sort_3d_heap
func.func @sparse_sort_3d_heap(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort heap_sort %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_partition_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_qsort_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-LABEL:   func.func @sparse_sort_coo_quick
func.func @sparse_sort_coo_quick(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo quick_sort %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_binary_search_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-DAG:     func.func private @_sparse_shift_down_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>, %arg5: index) {
// CHECK-DAG:     func.func private @_sparse_heap_sort_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-DAG:     func.func private @_sparse_partition_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_hybrid_qsort_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>, %arg5: i64) {
// CHECK-LABEL:   func.func @sparse_sort_coo_hybrid
func.func @sparse_sort_coo_hybrid(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo hybrid_quick_sort %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_binary_search_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-LABEL:   func.func @sparse_sort_coo_stable
func.func @sparse_sort_coo_stable(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo insertion_sort_stable %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_shift_down_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>, %arg5: index) {
// CHECK-DAG:     func.func private @_sparse_heap_sort_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-LABEL:   func.func @sparse_sort_coo_heap
func.func @sparse_sort_coo_heap(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo heap_sort %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}
