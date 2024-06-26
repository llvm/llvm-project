// RUN: mlir-opt %s --stage-sparse-ops --lower-sparse-ops-to-foreach --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : dense, d0 : compressed, d1 : compressed)
}>

// CHECK-LABEL:   func.func @sparse_convert_1d
// CHECK:           sparse_tensor.foreach
// CHECK:            scf.if
// CHECK:              tensor.insert
// CHECK-NOT:       sparse_tensor.reorder_coo
// CHECK:           sparse_tensor.load
func.func @sparse_convert_1d(%arg0: tensor<?xi32>) -> tensor<?xi32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xi32> to tensor<?xi32, #SparseVector>
  return %0 : tensor<?xi32, #SparseVector>
}

// CHECK-LABEL:   func.func @sparse_convert_complex
// CHECK:           sparse_tensor.foreach
// CHECK:            scf.if
// CHECK:              tensor.insert
// CHECK-NOT:       sparse_tensor.reorder_coo
// CHECK:           sparse_tensor.load
func.func @sparse_convert_complex(%arg0: tensor<100xcomplex<f64>>) -> tensor<100xcomplex<f64>, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<100xcomplex<f64>> to tensor<100xcomplex<f64>, #SparseVector>
  return %0 : tensor<100xcomplex<f64>, #SparseVector>
}

// CHECK-LABEL:   func.func @sparse_convert_2d
// CHECK:           sparse_tensor.foreach
// CHECK:            scf.if
// CHECK:              tensor.insert
// CHECK-NOT:       sparse_tensor.reorder_coo
// CHECK:           sparse_tensor.load
func.func @sparse_convert_2d(%arg0: tensor<2x4xf64>) -> tensor<2x4xf64, #CSR> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x4xf64> to tensor<2x4xf64, #CSR>
  return %0 : tensor<2x4xf64, #CSR>
}

// CHECK-LABEL:   func.func @sparse_constant
// CHECK:           sparse_tensor.foreach
// CHECK-NOT:         scf.if
// CHECK:               tensor.insert
// CHECK-NOT:       sparse_tensor.reorder_coo
// CHECK:           sparse_tensor.load
func.func @sparse_constant() -> tensor<8x7xf32, #CSR>{
  // Initialize a tensor.
  %0 = arith.constant sparse<[[0, 0], [1, 6]], [1.0, 5.0]> : tensor<8x7xf32>
  // Convert the tensor to a sparse tensor.
  %1 = sparse_tensor.convert %0 : tensor<8x7xf32> to tensor<8x7xf32, #CSR>
  return %1 : tensor<8x7xf32, #CSR>
}

// CHECK-LABEL:   func.func @sparse_constant_csc
// CHECK:           sparse_tensor.foreach
// CHECK-NOT:         scf.if
// CHECK:               tensor.insert
// CHECK-NOT:       sparse_tensor.reorder_coo
// CHECK:           sparse_tensor.load
func.func @sparse_constant_csc() -> tensor<8x7xf32, #CSC>{
  // Initialize a tensor.
  %0 = arith.constant sparse<[[0, 0], [1, 6]], [1.0, 5.0]> : tensor<8x7xf32>
  // Convert the tensor to a sparse tensor.
  %1 = sparse_tensor.convert %0 : tensor<8x7xf32> to tensor<8x7xf32, #CSC>
  return %1 : tensor<8x7xf32, #CSC>
}

// CHECK-LABEL:   func.func @sparse_convert_3d
// CHECK:           sparse_tensor.foreach
// CHECK:             scf.if
// CHECK:               tensor.insert
// CHECK:           sparse_tensor.load
// CHECK:           %[[TMP:.*]] = sparse_tensor.reorder_coo
// CHECK:           sparse_tensor.foreach
// CHECK:             tensor.insert
// CHECK:           sparse_tensor.load
// CHECK:           bufferization.dealloc_tensor %[[TMP]]
func.func @sparse_convert_3d(%arg0: tensor<?x?x?xf64>) -> tensor<?x?x?xf64, #SparseTensor> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf64> to tensor<?x?x?xf64, #SparseTensor>
  return %0 : tensor<?x?x?xf64, #SparseTensor>
}
