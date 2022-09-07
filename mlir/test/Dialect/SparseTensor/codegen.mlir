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

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i, j) -> (j, i)>
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
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xf64>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]] : memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_multi_ret(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<1xindex>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi32>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xi64>,
//  CHECK-SAME: %[[A7:.*7]]: memref<?xf64>) ->
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]]
func.func @sparse_nop_multi_ret(%arg0: tensor<?xf64, #SparseVector>,
                                %arg1: tensor<?xf64, #SparseVector>) ->
                                (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) {
  return %arg0, %arg1 : tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_call(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<1xindex>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi32>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xi64>,
//  CHECK-SAME: %[[A7:.*7]]: memref<?xf64>) 
//       CHECK: %[[T0:.*]]:8 = call @sparse_nop_multi_ret(%[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]]) 
//       CHECK: return %[[T0]]#0, %[[T0]]#1, %[[T0]]#2, %[[T0]]#3, %[[T0]]#4, %[[T0]]#5, %[[T0]]#6, %[[T0]]#7 
func.func @sparse_nop_call(%arg0: tensor<?xf64, #SparseVector>,
                           %arg1: tensor<?xf64, #SparseVector>) ->
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) {
  %1, %2 = call @sparse_nop_multi_ret(%arg0, %arg1) :
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) ->
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
  return %1, %2: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>
}

//
// CHECK-LABEL: func @sparse_nop_cast(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xf32>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]] : memref<1xindex>, memref<?xi32>, memref<?xi64>, memref<?xf32>
func.func @sparse_nop_cast(%arg0: tensor<64xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %arg0 : tensor<64xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

//
// CHECK-LABEL: func @sparse_nop_cast_3d(
//  CHECK-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xf32>)
//       CHECK: return %[[A0]], %[[A1]] : memref<3xindex>, memref<?xf32>
func.func @sparse_nop_cast_3d(%arg0: tensor<10x20x30xf32, #Dense3D>) -> tensor<?x?x?xf32, #Dense3D> {
  %0 = tensor.cast %arg0 : tensor<10x20x30xf32, #Dense3D> to tensor<?x?x?xf32, #Dense3D>
  return %0 : tensor<?x?x?xf32, #Dense3D>
}

//
// CHECK-LABEL: func @sparse_dense_2d(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xf64>) {
//       CHECK: return
func.func @sparse_dense_2d(%arg0: tensor<?x?xf64, #Dense2D>) {
  return
}

//
// CHECK-LABEL: func @sparse_row(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xf64>) {
//       CHECK: return
func.func @sparse_row(%arg0: tensor<?x?xf64, #Row>) {
  return
}

//
// CHECK-LABEL: func @sparse_csr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xf64>) {
//       CHECK: return
func.func @sparse_csr(%arg0: tensor<?x?xf64, #CSR>) {
  return
}

//
// CHECK-LABEL: func @sparse_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi32>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xf64>) {
//       CHECK: return
func.func @sparse_dcsr(%arg0: tensor<?x?xf64, #DCSR>) {
  return
}

//
// Querying for dimension 1 in the tensor type can immediately
// fold using the original static dimension sizes.
//
//
// CHECK-LABEL: func @sparse_dense_3d(
//  CHECK-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xf64>)
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
//
// CHECK-LABEL: func @sparse_dense_3d_dyn(
//  CHECK-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xf64>)
//       CHECK: %[[C:.*]] = arith.constant 2 : index
//       CHECK: %[[L:.*]] = memref.load %[[A0]][%[[C]]] : memref<3xindex>
//       CHECK: return %[[L]] : index
func.func @sparse_dense_3d_dyn(%arg0: tensor<?x?x?xf64, #Dense3D>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<?x?x?xf64, #Dense3D>
  return %0 : index
}

//
// CHECK-LABEL: func @sparse_pointers_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi32>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xf64>)
//       CHECK: return %[[A3]] : memref<?xi32>
func.func @sparse_pointers_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi32> {
  %c = arith.constant 1 : index
  %0 = sparse_tensor.pointers %arg0, %c : tensor<?x?xf64, #DCSR> to memref<?xi32>
  return %0 : memref<?xi32>
}

//
// CHECK-LABEL: func @sparse_indices_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi32>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xf64>)
//       CHECK: return %[[A4]] : memref<?xi64>
func.func @sparse_indices_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi64> {
  %c = arith.constant 1 : index
  %0 = sparse_tensor.indices %arg0, %c : tensor<?x?xf64, #DCSR> to memref<?xi64>
  return %0 : memref<?xi64>
}

//
// CHECK-LABEL: func @sparse_values_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi32>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xf64>)
//       CHECK: return %[[A5]] : memref<?xf64>
func.func @sparse_values_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #DCSR> to memref<?xf64>
  return %0 : memref<?xf64>
}

//
// CHECK-LABEL: func @sparse_dealloc_csr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xf64>) {
//       CHECK: memref.dealloc %[[A0]] : memref<2xindex>
//       CHECK: memref.dealloc %[[A1]] : memref<?xi32>
//       CHECK: memref.dealloc %[[A2]] : memref<?xi64>
//       CHECK: memref.dealloc %[[A3]] : memref<?xf64>
//       CHECK: return
func.func @sparse_dealloc_csr(%arg0: tensor<?x?xf64, #CSR>) {
  bufferization.dealloc_tensor %arg0 : tensor<?x?xf64, #CSR>
  return
}

//        CHECK-LABEL: func @sparse_alloc_csc(
//         CHECK-SAME: %[[A:.*]]: index) ->
//         CHECK-SAME: memref<2xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>
//          CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//          CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//          CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
//              CHECK: %[[T0:.*]] = memref.alloc() : memref<2xindex>
//              CHECK: memref.store %[[A]], %[[T0]][%[[C0]]] : memref<2xindex>
//              CHECK: memref.store %[[C10]], %[[T0]][%[[C1]]] : memref<2xindex>
//              CHECK: %[[T1:.*]] = memref.alloc() : memref<1xindex>
//              CHECK: %[[T2:.*]] = memref.cast %[[T1]] : memref<1xindex> to memref<?xindex>
//              CHECK: %[[T3:.*]] = memref.alloc() : memref<1xindex>
//              CHECK: %[[T4:.*]] = memref.cast %[[T3]] : memref<1xindex> to memref<?xindex>
//              CHECK: %[[T5:.*]] = memref.alloc() : memref<1xf64>
//              CHECK: %[[T6:.*]] = memref.cast %[[T5]] : memref<1xf64> to memref<?xf64>
//              CHECK: return %[[T0]], %[[T2]], %[[T4]], %[[T6]]
func.func @sparse_alloc_csc(%arg0: index) -> tensor<10x?xf64, #CSC> {
  %0 = bufferization.alloc_tensor(%arg0) : tensor<10x?xf64, #CSC>
  %1 = sparse_tensor.load %0 : tensor<10x?xf64, #CSC>
  return %1 : tensor<10x?xf64, #CSC>
}

//        CHECK-LABEL: func @sparse_alloc_3d() ->
//         CHECK-SAME: memref<3xindex>, memref<?xf64>
//          CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//          CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//          CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//          CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
//          CHECK-DAG: %[[C20:.*]] = arith.constant 20 : index
//          CHECK-DAG: %[[C30:.*]] = arith.constant 30 : index
//              CHECK: %[[A0:.*]] = memref.alloc() : memref<3xindex>
//              CHECK: memref.store %[[C30]], %[[A0]][%[[C0]]] : memref<3xindex>
//              CHECK: memref.store %[[C10]], %[[A0]][%[[C1]]] : memref<3xindex>
//              CHECK: memref.store %[[C20]], %[[A0]][%[[C2]]] : memref<3xindex>
//              CHECK: %[[A:.*]] = memref.alloc() : memref<6000xf64>
//              CHECK: %[[A1:.*]] = memref.cast %[[A]] : memref<6000xf64> to memref<?xf64>
//              CHECK: return %[[A0]], %[[A1]] : memref<3xindex>, memref<?xf64>
func.func @sparse_alloc_3d() -> tensor<10x20x30xf64, #Dense3D> {
  %0 = bufferization.alloc_tensor() : tensor<10x20x30xf64, #Dense3D>
  %1 = sparse_tensor.load %0 : tensor<10x20x30xf64, #Dense3D>
  return %1 : tensor<10x20x30xf64, #Dense3D>
}
