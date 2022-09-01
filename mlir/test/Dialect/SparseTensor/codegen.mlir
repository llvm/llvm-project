// RUN: mlir-opt %s --sparse-tensor-codegen  --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  indexBitWidth = 64,
  pointerBitWidth = 32
}>

#Dense2D = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense" ],
  indexBitWidth = 64,
  pointerBitWidth = 32
}>

#Row = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense" ],
  indexBitWidth = 64,
  pointerBitWidth = 32
}>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  indexBitWidth = 64,
  pointerBitWidth = 32
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  indexBitWidth = 64,
  pointerBitWidth = 32
}>

#Dense3D = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense", "dense" ],
  indexBitWidth = 64,
  pointerBitWidth = 32,
  dimOrdering = affine_map<(i,j,k) -> (k, i,j)>
}>

// CHECK-LABEL: func @sparse_nop(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<1xindex>, memref<?xi64>, memref<?xi32>, memref<?xf64>>) -> tuple<memref<1xindex>, memref<?xi64>, memref<?xi32>, memref<?xf64>>
//       CHECK: return %[[A]] : tuple<memref<1xindex>, memref<?xi64>, memref<?xi32>, memref<?xf64>>
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_dense_2d(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xf64>>)
func.func @sparse_dense_2d(%arg0: tensor<?x?xf64, #Dense2D>) {
  return
}

// CHECK-LABEL: func @sparse_row(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi64>, memref<?xi32>, memref<?xf64>>)
func.func @sparse_row(%arg0: tensor<?x?xf64, #Row>) {
  return
}

// CHECK-LABEL: func @sparse_csr(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi64>, memref<?xi32>, memref<?xf64>>)
func.func @sparse_csr(%arg0: tensor<?x?xf64, #CSR>) {
  return
}

// CHECK-LABEL: func @sparse_dcsr(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xf64>>)
func.func @sparse_dcsr(%arg0: tensor<?x?xf64, #DCSR>) {
  return
}

//
// Just a linearized array in the end. Dim op is statically known.
//
// CHECK-LABEL: func @sparse_dense_3d(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<6000xf64>>) -> index
//       CHECK: %[[C:.*]] = arith.constant 20 : index
//       CHECK: return %[[C]] : index
func.func @sparse_dense_3d(%arg0: tensor<10x20x30xf64, #Dense3D>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<10x20x30xf64, #Dense3D>
  return %0 : index
}
