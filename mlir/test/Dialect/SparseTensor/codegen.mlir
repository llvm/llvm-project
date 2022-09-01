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
  dimOrdering = affine_map<(i, j, k) -> (k, i, j)>
}>

// CHECK-LABEL: func @sparse_nop(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>) -> tuple<memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>
//       CHECK: return %[[A]] : tuple<memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_dense_2d(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xf64>>)
func.func @sparse_dense_2d(%arg0: tensor<?x?xf64, #Dense2D>) {
  return
}

// CHECK-LABEL: func @sparse_row(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
func.func @sparse_row(%arg0: tensor<?x?xf64, #Row>) {
  return
}

// CHECK-LABEL: func @sparse_csr(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
func.func @sparse_csr(%arg0: tensor<?x?xf64, #CSR>) {
  return
}

// CHECK-LABEL: func @sparse_dcsr(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
func.func @sparse_dcsr(%arg0: tensor<?x?xf64, #DCSR>) {
  return
}

//
// Querying for dimension 1 in the tensor type can immediately
// fold using the original static dimension sizes.
//
// CHECK-LABEL: func @sparse_dense_3d(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<3xindex>, memref<6000xf64>>) -> index {
//       CHECK: %[[C:.*]] = arith.constant 20 : index
//       CHECK: return %[[C]] : index
func.func @sparse_dense_3d(%arg0: tensor<10x20x30xf64, #Dense3D>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<10x20x30xf64, #Dense3D>
  return %0 : index
}

//
// Querying for dimension 1 in the tensor type needs to be permuted
// into querying for dimension 2 in the stored sparse tensor scheme,
// since the latter honors the dimOrdering.
//
// CHECK-LABEL: func @sparse_dense_3d_dyn(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<3xindex>, memref<?xf64>>) -> index {
//       CHECK: %[[C:.*]] = arith.constant 2 : index
//       CHECK: %[[F:.*]] = sparse_tensor.storage_get %[[A]][0] : tuple<memref<3xindex>, memref<?xf64>> to memref<3xindex>
//       CHECK: %[[L:.*]] = memref.load %[[F]][%[[C]]] : memref<3xindex>
//       CHECK: return %[[L]] : index
func.func @sparse_dense_3d_dyn(%arg0: tensor<?x?x?xf64, #Dense3D>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<?x?x?xf64, #Dense3D>
  return %0 : index
}

// CHECK-LABEL: func @sparse_pointers_dcsr(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//       CHECK: %[[F:.*]] = sparse_tensor.storage_get %[[A]][3] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xi32>
//       CHECK: return %[[F]] : memref<?xi32>
func.func @sparse_pointers_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi32> {
  %c = arith.constant 1 : index
  %0 = sparse_tensor.pointers %arg0, %c : tensor<?x?xf64, #DCSR> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_indices_dcsr(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//       CHECK: %[[F:.*]] = sparse_tensor.storage_get %[[A]][4] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xi64>
//       CHECK: return %[[F]] : memref<?xi64>
func.func @sparse_indices_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi64> {
  %c = arith.constant 1 : index
  %0 = sparse_tensor.indices %arg0, %c : tensor<?x?xf64, #DCSR> to memref<?xi64>
  return %0 : memref<?xi64>
}

// CHECK-LABEL: func @sparse_values_dcsr(
//  CHECK-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//       CHECK: %[[F:.*]] = sparse_tensor.storage_get %[[A]][5] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xf64>
//       CHECK: return %[[F]] : memref<?xf64>
func.func @sparse_values_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #DCSR> to memref<?xf64>
  return %0 : memref<?xf64>
}
