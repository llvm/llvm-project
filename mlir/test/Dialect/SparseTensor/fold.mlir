// RUN: mlir-opt %s  --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_nop_convert(
//  CHECK-SAME: %[[A:.*]]: tensor<64xf32, #{{.*}}>)
//       CHECK: return %[[A]] : tensor<64xf32, #{{.*}}>
func @sparse_nop_convert(%arg0: tensor<64xf32, #SparseVector>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32, #SparseVector> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_dce_convert(
//  CHECK-SAME: %[[A:.*]]: tensor<64xf32>)
//   CHECK-NOT: sparse_tensor.convert
//       CHECK: return
func @sparse_dce_convert(%arg0: tensor<64xf32>) {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32> to tensor<64xf32, #SparseVector>
  return
}
