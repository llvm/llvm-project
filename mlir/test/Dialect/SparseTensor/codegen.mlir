// RUN: mlir-opt %s --sparse-tensor-codegen  --canonicalize --cse | FileCheck %s --check-prefix=CHECK-CODEGEN
// RUN: mlir-opt %s --sparse-tensor-codegen --sparse-tensor-storage-expansion --canonicalize --cse | FileCheck %s --check-prefix=CHECK-STORAGE


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

// CHECK-CODEGEN-LABEL: func @sparse_nop(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//       CHECK-CODEGEN: return %[[A]] : tuple<memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>
//
// CHECK-STORAGE-LABEL: func @sparse_nop(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xf64>) 
//       CHECK-STORAGE: return %[[A0]], %[[A1]], %[[A2]], %[[A3]] : memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}

// CHECK-CODEGEN-LABEL: func @sparse_nop_cast(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf32>>)
//       CHECK-CODEGEN: return %[[A]] : tuple<memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf32>>
//
// CHECK-STORAGE-LABEL: func @sparse_nop_cast(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xf32>) 
//       CHECK-STORAGE: return %[[A0]], %[[A1]], %[[A2]], %[[A3]] : memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf32>
func.func @sparse_nop_cast(%arg0: tensor<64xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %arg0 : tensor<64xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-CODEGEN-LABEL: func @sparse_nop_cast_3d(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<3xindex>, memref<?xf32>>)
//       CHECK-CODEGEN: return %[[A]] : tuple<memref<3xindex>, memref<?xf32>>
//
// CHECK-STORAGE-LABEL: func @sparse_nop_cast_3d(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xf32>) 
//       CHECK-STORAGE: return %[[A0]], %[[A1]] : memref<3xindex>, memref<?xf32>
func.func @sparse_nop_cast_3d(%arg0: tensor<10x20x30xf32, #Dense3D>) -> tensor<?x?x?xf32, #Dense3D> {
  %0 = tensor.cast %arg0 : tensor<10x20x30xf32, #Dense3D> to tensor<?x?x?xf32, #Dense3D>
  return %0 : tensor<?x?x?xf32, #Dense3D>
}

// CHECK-CODEGEN-LABEL: func @sparse_dense_2d(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xf64>>)
//
// CHECK-STORAGE-LABEL: func @sparse_dense_2d(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xf64>) {
//       CHECK-STORAGE: return
func.func @sparse_dense_2d(%arg0: tensor<?x?xf64, #Dense2D>) {
  return
}

// CHECK-CODEGEN-LABEL: func @sparse_row(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//
// CHECK-STORAGE-LABEL: func @sparse_row(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xf64>) {
//       CHECK-STORAGE: return
func.func @sparse_row(%arg0: tensor<?x?xf64, #Row>) {
  return
}

// CHECK-CODEGEN-LABEL: func @sparse_csr(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//
// CHECK-STORAGE-LABEL: func @sparse_csr(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xf64>) {
//       CHECK-STORAGE: return
func.func @sparse_csr(%arg0: tensor<?x?xf64, #CSR>) {
  return
}

// CHECK-CODEGEN-LABEL: func @sparse_dcsr(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//
// CHECK-STORAGE-LABEL: func @sparse_dcsr(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A4:.*4]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A5:.*5]]: memref<?xf64>) {
//       CHECK-STORAGE: return
func.func @sparse_dcsr(%arg0: tensor<?x?xf64, #DCSR>) {
  return
}

//
// Querying for dimension 1 in the tensor type can immediately
// fold using the original static dimension sizes.
//
// CHECK-CODEGEN-LABEL: func @sparse_dense_3d(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<3xindex>, memref<?xf64>>)
//       CHECK-CODEGEN: %[[C:.*]] = arith.constant 20 : index
//       CHECK-CODEGEN: return %[[C]] : index
//
// CHECK-STORAGE-LABEL: func @sparse_dense_3d(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xf64>) 
//       CHECK-STORAGE: %[[C:.*]] = arith.constant 20 : index
//       CHECK-STORAGE: return %[[C]] : index
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
// CHECK-CODEGEN-LABEL: func @sparse_dense_3d_dyn(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<3xindex>, memref<?xf64>>)
//       CHECK-CODEGEN: %[[C:.*]] = arith.constant 2 : index
//       CHECK-CODEGEN: %[[F:.*]] = sparse_tensor.storage_get %[[A]][0] : tuple<memref<3xindex>, memref<?xf64>> to memref<3xindex>
//       CHECK-CODEGEN: %[[L:.*]] = memref.load %[[F]][%[[C]]] : memref<3xindex>
//       CHECK-CODEGEN: return %[[L]] : index
//
// CHECK-STORAGE-LABEL: func @sparse_dense_3d_dyn(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xf64>) 
//       CHECK-STORAGE: %[[C:.*]] = arith.constant 2 : index
//       CHECK-STORAGE: %[[L:.*]] = memref.load %[[A0]][%[[C]]] : memref<3xindex>
//       CHECK-STORAGE: return %[[L]] : index
func.func @sparse_dense_3d_dyn(%arg0: tensor<?x?x?xf64, #Dense3D>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<?x?x?xf64, #Dense3D>
  return %0 : index
}

// CHECK-CODEGEN-LABEL: func @sparse_pointers_dcsr(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//       CHECK-CODEGEN: %[[F:.*]] = sparse_tensor.storage_get %[[A]][3] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xi32>
//       CHECK-CODEGEN: return %[[F]] : memref<?xi32>
//
// CHECK-STORAGE-LABEL: func @sparse_pointers_dcsr(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A4:.*4]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A5:.*5]]: memref<?xf64>) 
//       CHECK-STORAGE: return %[[A3]] : memref<?xi32>
func.func @sparse_pointers_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi32> {
  %c = arith.constant 1 : index
  %0 = sparse_tensor.pointers %arg0, %c : tensor<?x?xf64, #DCSR> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-CODEGEN-LABEL: func @sparse_indices_dcsr(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//       CHECK-CODEGEN: %[[F:.*]] = sparse_tensor.storage_get %[[A]][4] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xi64>
//       CHECK-CODEGEN: return %[[F]] : memref<?xi64>
//
// CHECK-STORAGE-LABEL: func @sparse_indices_dcsr(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A4:.*4]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A5:.*5]]: memref<?xf64>) 
//       CHECK-STORAGE: return %[[A4]] : memref<?xi64>
func.func @sparse_indices_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi64> {
  %c = arith.constant 1 : index
  %0 = sparse_tensor.indices %arg0, %c : tensor<?x?xf64, #DCSR> to memref<?xi64>
  return %0 : memref<?xi64>
}

// CHECK-CODEGEN-LABEL: func @sparse_values_dcsr(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//       CHECK-CODEGEN: %[[F:.*]] = sparse_tensor.storage_get %[[A]][5] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xf64>
//       CHECK-CODEGEN: return %[[F]] : memref<?xf64>
//
// CHECK-STORAGE-LABEL: func @sparse_values_dcsr(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A4:.*4]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A5:.*5]]: memref<?xf64>) 
//       CHECK-STORAGE: return %[[A5]] : memref<?xf64>
func.func @sparse_values_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #DCSR> to memref<?xf64>
  return %0 : memref<?xf64>
}

// CHECK-CODEGEN-LABEL: func @sparse_dealloc_csr(
//  CHECK-CODEGEN-SAME: %[[A:.*]]: tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>>)
//       CHECK-CODEGEN: %[[F0:.*]] = sparse_tensor.storage_get %[[A]][0] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<2xindex>
//       CHECK-CODEGEN: memref.dealloc %[[F0]] : memref<2xindex>
//       CHECK-CODEGEN: %[[F1:.*]] = sparse_tensor.storage_get %[[A]][1] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xi32>
//       CHECK-CODEGEN: memref.dealloc %[[F1]] : memref<?xi32>
//       CHECK-CODEGEN: %[[F2:.*]] = sparse_tensor.storage_get %[[A]][2] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xi64>
//       CHECK-CODEGEN: memref.dealloc %[[F2]] : memref<?xi64>
//       CHECK-CODEGEN: %[[F3:.*]] = sparse_tensor.storage_get %[[A]][3] : tuple<memref<2xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>> to memref<?xf64>
//       CHECK-CODEGEN: memref.dealloc %[[F3]] : memref<?xf64>
//       CHECK-CODEGEN: return
//
// CHECK-STORAGE-LABEL: func @sparse_dealloc_csr(
//  CHECK-STORAGE-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-STORAGE-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-STORAGE-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-STORAGE-SAME: %[[A3:.*3]]: memref<?xf64>) {
//       CHECK-STORAGE: memref.dealloc %[[A0]] : memref<2xindex>
//       CHECK-STORAGE: memref.dealloc %[[A1]] : memref<?xi32>
//       CHECK-STORAGE: memref.dealloc %[[A2]] : memref<?xi64>
//       CHECK-STORAGE: memref.dealloc %[[A3]] : memref<?xf64>
//       CHECK-STORAGE: return
func.func @sparse_dealloc_csr(%arg0: tensor<?x?xf64, #CSR>) {
  bufferization.dealloc_tensor %arg0 : tensor<?x?xf64, #CSR>
  return
}
