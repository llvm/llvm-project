// RUN: mlir-opt %s --stage-sparse-ops --post-sparsification-rewrite="enable-foreach=false" --canonicalize --cse | FileCheck %s

#SparseVector64 = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed),
  posWidth = 64,
  crdWidth = 64
}>

#SparseVector32 = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#SortedCOO2D = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton),
}>

#SortedCOO3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique), d2 : singleton)

}>

#TsssPermuted = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : compressed, d0 : compressed, d1 : compressed)
}>

#COOSlice = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(2, 2, 1)>, d1 : #sparse_tensor<slice(12, 13, 1)>) -> (d0 : compressed(nonunique), d1 : singleton)
}>

// CHECK-LABEL:   func.func @sparse_nop_convert
// CHECK-NEXT:       return
func.func @sparse_nop_convert(%arg0: tensor<64xf32, #SparseVector>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32, #SparseVector> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// CHECK-LABEL:   func.func @sparse_hidden_nop_cast
// TODO: The following convert should be a cast instead.
// CHECK:           sparse_tensor.convert
// CHECK:           return
func.func @sparse_hidden_nop_cast(%arg0: tensor<32xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<32xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL:   func.func @sparse_convert_1d_ss(
// TODO: libgen path need to support efficient format conversion (e.g., 32 bit pos -> 64 bit pos).
// Maybe we should use a different operator as well to be clear.
func.func @sparse_convert_1d_ss(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

// CHECK-LABEL:   func.func @sparse_convert(
// TODO: libgen path need to support efficient format conversion (e.g., 32 bit pos -> 64 bit pos).
// Maybe we should use a different operator as well to be clear.
func.func @sparse_convert(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

// CHECK-LABEL:   func.func @sparse_convert_permuted
// CHECK:           sparse_tensor.foreach
// CHECK:             sparse_tensor.insert
// CHECK:           sparse_tensor.load
// CHECK:           sparse_tensor.reorder_coo
// CHECK:           sparse_tensor.foreach
// CHECK:             sparse_tensor.insert
// CHECK:           sparse_tensor.load
func.func @sparse_convert_permuted(%arg0: tensor<?x?x?xf32, #SortedCOO3D>) -> tensor<?x?x?xf32, #TsssPermuted> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf32, #SortedCOO3D> to tensor<?x?x?xf32, #TsssPermuted>
  return %0 : tensor<?x?x?xf32, #TsssPermuted>
}

// CHECK-LABEL:   func.func @sparse_convert_slice
// CHECK:           sparse_tensor.foreach
// CHECK:             sparse_tensor.insert
// CHECK:           sparse_tensor.load
// CHECK-NOT:       sparse_tensor.reorder_coo
func.func @sparse_convert_slice(%arg0: tensor<2x13xi32, #COOSlice>) -> (tensor<2x13xi32, #SortedCOO2D>)  {
  %0 = sparse_tensor.convert %arg0 : tensor<2x13xi32, #COOSlice> to tensor<2x13xi32, #SortedCOO2D>
  return %0 : tensor<2x13xi32, #SortedCOO2D>
}
