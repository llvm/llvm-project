// RUN: mlir-opt %s --stage-sparse-ops --post-sparsification-rewrite="enable-foreach=false" --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : dense, d0 : compressed, d1 : compressed)
}>

// CHECK-LABEL:  func.func @sparse_convert_1d
// CHECK-NOT:      sparse_tensor.reorder_coo
// CHECK:          bufferization.alloc_tensor
// CHECK:          linalg.fill
// CHECK:          sparse_tensor.foreach
// CHECK:            tensor.insert
func.func @sparse_convert_1d(%arg0: tensor<13xi32, #SparseVector>) -> tensor<13xi32> {
  %0 = sparse_tensor.convert %arg0 : tensor<13xi32, #SparseVector> to tensor<13xi32>
  return %0 : tensor<13xi32>
}

// CHECK-LABEL:  func.func @sparse_convert_1d_dyn
// CHECK-NOT:      sparse_tensor.reorder_coo
// CHECK:          bufferization.alloc_tensor
// CHECK:          linalg.fill
// CHECK:          sparse_tensor.foreach
// CHECK:            tensor.insert
func.func @sparse_convert_1d_dyn(%arg0: tensor<?xi32, #SparseVector>) -> tensor<?xi32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xi32, #SparseVector> to tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL:  func.func @sparse_convert_2d
// CHECK-NOT:      sparse_tensor.reorder_coo
// CHECK:          bufferization.alloc_tensor
// CHECK:          linalg.fill
// CHECK:          sparse_tensor.foreach
// CHECK:            tensor.insert
func.func @sparse_convert_2d(%arg0: tensor<2x4xf64, #SparseMatrix>) -> tensor<2x4xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x4xf64, #SparseMatrix> to tensor<2x4xf64>
  return %0 : tensor<2x4xf64>
}

// CHECK-LABEL:  func.func @sparse_convert_2d_dyn
// CHECK-NOT:      sparse_tensor.reorder_coo
// CHECK:          bufferization.alloc_tensor
// CHECK:          linalg.fill
// CHECK:          sparse_tensor.foreach
// CHECK:            tensor.insert
func.func @sparse_convert_2d_dyn0(%arg0: tensor<?x4xf64, #SparseMatrix>) -> tensor<?x4xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x4xf64, #SparseMatrix> to tensor<?x4xf64>
  return %0 : tensor<?x4xf64>
}

// CHECK-LABEL:  func.func @sparse_convert_2d_dyn1
// CHECK-NOT:      sparse_tensor.reorder_coo
// CHECK:          bufferization.alloc_tensor
// CHECK:          linalg.fill
// CHECK:          sparse_tensor.foreach
// CHECK:            tensor.insert
func.func @sparse_convert_2d_dyn1(%arg0: tensor<2x?xf64, #SparseMatrix>) -> tensor<2x?xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x?xf64, #SparseMatrix> to tensor<2x?xf64>
  return %0 : tensor<2x?xf64>
}

// CHECK-LABEL:  func.func @sparse_convert_2d_dyn2
// CHECK-NOT:      sparse_tensor.reorder_coo
// CHECK:          bufferization.alloc_tensor
// CHECK:          linalg.fill
// CHECK:          sparse_tensor.foreach
// CHECK:            tensor.insert
func.func @sparse_convert_2d_dyn2(%arg0: tensor<?x?xf64, #SparseMatrix>) -> tensor<?x?xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?xf64, #SparseMatrix> to tensor<?x?xf64>
  return %0 : tensor<?x?xf64>
}

// CHECK-LABEL:  func.func @sparse_convert_3d
// CHECK-NOT:      sparse_tensor.reorder_coo
// CHECK:          bufferization.alloc_tensor
// CHECK:          linalg.fill
// CHECK:          sparse_tensor.foreach
// CHECK:            tensor.insert
func.func @sparse_convert_3d(%arg0: tensor<2x3x4xf64, #SparseTensor>) -> tensor<2x3x4xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x3x4xf64, #SparseTensor> to tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}
