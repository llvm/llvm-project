// RUN: mlir-opt %s  --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_nop_convert(
//  CHECK-SAME: %[[A:.*]]: tensor<64xf32, #sparse_tensor.encoding<{{{.*}}}>>)
//   CHECK-NOT: sparse_tensor.convert
//       CHECK: return %[[A]] : tensor<64xf32, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_nop_convert(%arg0: tensor<64xf32, #SparseVector>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32, #SparseVector> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_dce_convert(
//  CHECK-SAME: %[[A:.*]]: tensor<64xf32>)
//   CHECK-NOT: sparse_tensor.convert
//       CHECK: return
func.func @sparse_dce_convert(%arg0: tensor<64xf32>) {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32> to tensor<64xf32, #SparseVector>
  return
}

// CHECK-LABEL: func @sparse_dce_getters(
//  CHECK-SAME: %[[A:.*]]: tensor<64xf32, #sparse_tensor.encoding<{{{.*}}}>>)
//   CHECK-NOT: sparse_tensor.pointers
//   CHECK-NOT: sparse_tensor.indices
//   CHECK-NOT: sparse_tensor.values
//       CHECK: return
func.func @sparse_dce_getters(%arg0: tensor<64xf32, #SparseVector>) {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.pointers %arg0, %c : tensor<64xf32, #SparseVector> to memref<?xindex>
  %1 = sparse_tensor.indices %arg0, %c : tensor<64xf32, #SparseVector> to memref<?xindex>
  %2 = sparse_tensor.values %arg0 : tensor<64xf32, #SparseVector> to memref<?xf32>
  return
}
