// RUN: mlir-opt %s -sparse-tensor-storage-expansion | FileCheck %s

// CHECK-LABEL:  func @sparse_storage_expand(
// CHECK-SAME:     %[[TMP_arg0:.*0]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg1:.*1]]: memref<?xf64>,
// CHECK-SAME:     %[[TMP_arg2:.*]]: f64
// CHECK           return %[[TMP_arg0]], %[[TMP_arg1]], %[[TMP_arg2]]
func.func @sparse_storage_expand(%arg0: tuple<memref<?xf64>, memref<?xf64>, f64>)
                                     -> tuple<memref<?xf64>, memref<?xf64>, f64> {
  return %arg0 : tuple<memref<?xf64>, memref<?xf64>, f64>
}
