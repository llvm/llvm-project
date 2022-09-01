// RUN: mlir-opt %s -sparse-tensor-storage-expansion -cse | FileCheck %s

// CHECK-LABEL:  func @sparse_storage_expand(
// CHECK-SAME:     %[[TMP_arg0:.*0]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg1:.*1]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg2:.*]]: f64
// CHECK           return %[[TMP_arg0]], %[[TMP_arg1]], %[[TMP_arg2]]
func.func @sparse_storage_expand(%arg0: tuple<memref<?xf64>, memref<?xf64>, f64>)
                                     -> tuple<memref<?xf64>, memref<?xf64>, f64> {
  return %arg0 : tuple<memref<?xf64>, memref<?xf64>, f64>
}

// CHECK-LABEL:  func @call_sparse_storage_expand(
// CHECK-SAME:     %[[TMP_arg0:.*0]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg1:.*1]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg2:.*]]: f64) 
// CHECK:          %[[TMP_0:.*]]:3 = call @sparse_storage_expand(%[[TMP_arg0]], %[[TMP_arg1]], %[[TMP_arg2]]) 
// CHECK:          return %[[TMP_0]]#0, %[[TMP_0]]#1, %[[TMP_0]]#2 : memref<?xf64>, memref<?xf64>, f64
func.func @call_sparse_storage_expand(%arg0: tuple<memref<?xf64>, memref<?xf64>, f64>)
                                          -> tuple<memref<?xf64>, memref<?xf64>, f64> {
  %1 = call @sparse_storage_expand(%arg0) : (tuple<memref<?xf64>, memref<?xf64>, f64>) ->
                                             tuple<memref<?xf64>, memref<?xf64>, f64>
  return %1 : tuple<memref<?xf64>, memref<?xf64>, f64>
}

// CHECK-LABEL:  func @sparse_storage_get(
// CHECK-SAME:     %[[TMP_arg0:.*0]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg1:.*1]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg2:.*]]: f64) 
// CHECK:          return %[[TMP_arg0]] : memref<?xf64>
func.func @sparse_storage_get(%arg0: tuple<memref<?xf64>, memref<?xf64>, f64>) -> memref<?xf64> {
  %0 = sparse_tensor.storage_get %arg0[0]
       : tuple<memref<?xf64>, memref<?xf64>, f64> to memref<?xf64>
  return %0 : memref<?xf64>
}

// CHECK-LABEL:  func @sparse_storage_set(
// CHECK-SAME:     %[[TMP_arg0:.*0]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg1:.*1]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg2:.*]]: f64,
// CHECK-SAME:     %[[TMP_arg3:.*]]: memref<?xf64>) 
// CHECK:          return %[[TMP_arg3]], %[[TMP_arg1]], %[[TMP_arg2]] : memref<?xf64>, memref<?xf64>, f64
func.func @sparse_storage_set(%arg0: tuple<memref<?xf64>, memref<?xf64>, f64>,
                              %arg1: memref<?xf64>) -> tuple<memref<?xf64>, memref<?xf64>, f64> {
  %0 = sparse_tensor.storage_set %arg0[0], %arg1
       : tuple<memref<?xf64>, memref<?xf64>, f64>, memref<?xf64> to
         tuple<memref<?xf64>, memref<?xf64>, f64>
  return %0 : tuple<memref<?xf64>, memref<?xf64>, f64>
}
