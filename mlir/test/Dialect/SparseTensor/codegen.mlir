// RUN: mlir-opt %s --sparse-tensor-codegen  --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

// TODO: just a dummy memref rewriting to get the ball rolling....

// CHECK-LABEL: func @sparse_nop(
//  CHECK-SAME: %[[A:.*]]: memref<?xf64>) -> memref<?xf64> {
//       CHECK: return %[[A]] : memref<?xf64>
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}
