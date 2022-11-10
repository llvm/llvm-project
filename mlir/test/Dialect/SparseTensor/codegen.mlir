// RUN: mlir-opt %s --sparse-tensor-codegen  --canonicalize --cse | FileCheck %s

#SV = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

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

#UCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed-no" ]
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
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]] :
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_multi_ret(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<1xindex>,
//  CHECK-SAME: %[[A6:.*6]]: memref<3xindex>,
//  CHECK-SAME: %[[A7:.*7]]: memref<?xi32>,
//  CHECK-SAME: %[[A8:.*8]]: memref<?xi64>,
//  CHECK-SAME: %[[A9:.*9]]: memref<?xf64>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[A8]], %[[A9]] :
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>,
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>
func.func @sparse_nop_multi_ret(%arg0: tensor<?xf64, #SparseVector>,
                                %arg1: tensor<?xf64, #SparseVector>) ->
                                (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) {
  return %arg0, %arg1 : tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_call(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<1xindex>,
//  CHECK-SAME: %[[A6:.*6]]: memref<3xindex>,
//  CHECK-SAME: %[[A7:.*7]]: memref<?xi32>,
//  CHECK-SAME: %[[A8:.*8]]: memref<?xi64>,
//  CHECK-SAME: %[[A9:.*9]]: memref<?xf64>)
//       CHECK: %[[T:.*]]:10 = call @sparse_nop_multi_ret(%[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[A8]], %[[A9]])
//       CHECK: return %[[T]]#0, %[[T]]#1, %[[T]]#2, %[[T]]#3, %[[T]]#4, %[[T]]#5, %[[T]]#6, %[[T]]#7, %[[T]]#8, %[[T]]#9 :
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>,
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>
func.func @sparse_nop_call(%arg0: tensor<?xf64, #SparseVector>,
                           %arg1: tensor<?xf64, #SparseVector>) ->
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) {
  %1, %2 = call @sparse_nop_multi_ret(%arg0, %arg1) :
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) ->
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
  return %1, %2: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_cast(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf32>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]] :
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf32>
func.func @sparse_nop_cast(%arg0: tensor<64xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %arg0 : tensor<64xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_cast_3d(
//  CHECK-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<1xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf32>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]] :
//  CHECK-SAME:   memref<3xindex>, memref<1xindex>, memref<?xf32>
func.func @sparse_nop_cast_3d(%arg0: tensor<10x20x30xf32, #Dense3D>) -> tensor<?x?x?xf32, #Dense3D> {
  %0 = tensor.cast %arg0 : tensor<10x20x30xf32, #Dense3D> to tensor<?x?x?xf32, #Dense3D>
  return %0 : tensor<?x?x?xf32, #Dense3D>
}

// CHECK-LABEL: func @sparse_dense_2d(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<1xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf64>)
//       CHECK: return
func.func @sparse_dense_2d(%arg0: tensor<?x?xf64, #Dense2D>) {
  return
}

// CHECK-LABEL: func @sparse_row(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>)
//       CHECK: return
func.func @sparse_row(%arg0: tensor<?x?xf64, #Row>) {
  return
}

// CHECK-LABEL: func @sparse_csr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>)
//       CHECK: return
func.func @sparse_csr(%arg0: tensor<?x?xf64, #CSR>) {
  return
}

// CHECK-LABEL: func @sparse_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<5xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi32>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xf64>)
//       CHECK: return
func.func @sparse_dcsr(%arg0: tensor<?x?xf64, #DCSR>) {
  return
}

//
// Querying for dimension 1 in the tensor type can immediately
// fold using the original static dimension sizes.
//
// CHECK-LABEL: func @sparse_dense_3d(
//  CHECK-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<1xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf64>)
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
//  CHECK-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<1xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf64>)
//       CHECK: %[[C:.*]] = arith.constant 2 : index
//       CHECK: %[[L:.*]] = memref.load %[[A0]][%[[C]]] : memref<3xindex>
//       CHECK: return %[[L]] : index
func.func @sparse_dense_3d_dyn(%arg0: tensor<?x?x?xf64, #Dense3D>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<?x?x?xf64, #Dense3D>
  return %0 : index
}

// CHECK-LABEL: func @sparse_pointers_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<5xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi32>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xf64>)
//       CHECK: return %[[A4]] : memref<?xi32>
func.func @sparse_pointers_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi32> {
  %0 = sparse_tensor.pointers %arg0 { dimension = 1 : index } : tensor<?x?xf64, #DCSR> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_indices_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<5xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi32>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xf64>)
//       CHECK: return %[[A5]] : memref<?xi64>
func.func @sparse_indices_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi64> {
  %0 = sparse_tensor.indices %arg0 { dimension = 1 : index } : tensor<?x?xf64, #DCSR> to memref<?xi64>
  return %0 : memref<?xi64>
}

// CHECK-LABEL: func @sparse_values_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<5xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi32>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xf64>)
//       CHECK: return %[[A6]] : memref<?xf64>
func.func @sparse_values_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #DCSR> to memref<?xf64>
  return %0 : memref<?xf64>
}

// CHECK-LABEL: func @sparse_noe(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>)
//       CHECK: %[[C2:.*]] = arith.constant 2 : index
//       CHECK: %[[NOE:.*]] = memref.load %[[A1]][%[[C2]]] : memref<3xindex>
//       CHECK: return %[[NOE]] : index
func.func @sparse_noe(%arg0: tensor<128xf64, #SparseVector>) -> index {
  %0 = sparse_tensor.number_of_entries %arg0 : tensor<128xf64, #SparseVector>
  return %0 : index
}

// CHECK-LABEL: func @sparse_dealloc_csr(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>)
//       CHECK: memref.dealloc %[[A0]] : memref<2xindex>
//       CHECK: memref.dealloc %[[A1]] : memref<3xindex>
//       CHECK: memref.dealloc %[[A2]] : memref<?xi32>
//       CHECK: memref.dealloc %[[A3]] : memref<?xi64>
//       CHECK: memref.dealloc %[[A4]] : memref<?xf64>
//       CHECK: return
func.func @sparse_dealloc_csr(%arg0: tensor<?x?xf64, #CSR>) {
  bufferization.dealloc_tensor %arg0 : tensor<?x?xf64, #CSR>
  return
}

// CHECK-LABEL: func @sparse_alloc_csc(
//  CHECK-SAME: %[[A:.*]]: index) ->
//  CHECK-SAME: memref<2xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
//       CHECK: %[[T0:.*]] = memref.alloc() : memref<2xindex>
//       CHECK: %[[T1:.*]] = memref.alloc() : memref<3xindex>
//       CHECK: %[[T2:.*]] = memref.alloc() : memref<16xindex>
//       CHECK: %[[T3:.*]] = memref.cast %[[T2]] : memref<16xindex> to memref<?xindex>
//       CHECK: %[[T4:.*]] = memref.alloc() : memref<16xindex>
//       CHECK: %[[T5:.*]] = memref.cast %[[T4]] : memref<16xindex> to memref<?xindex>
//       CHECK: %[[T6:.*]] = memref.alloc() : memref<16xf64>
//       CHECK: %[[T7:.*]] = memref.cast %[[T6]] : memref<16xf64> to memref<?xf64>
//       CHECK: linalg.fill ins(%[[C0]] : index) outs(%[[T1]] : memref<3xindex>)
//       CHECK: memref.store %[[A]], %[[T0]][%[[C0]]] : memref<2xindex>
//       CHECK: memref.store %[[C10]], %[[T0]][%[[C1]]] : memref<2xindex>
//       CHECK: %[[P0:.*]] = sparse_tensor.push_back %[[T1]], %[[T3]]
//       CHECK: %[[P1:.*]] = sparse_tensor.push_back %[[T1]], %[[P0]]
//       CHECK: return %[[T0]], %[[T1]], %[[P1]], %[[T5]], %[[T7]] :
//  CHECK-SAME:   memref<2xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>
func.func @sparse_alloc_csc(%arg0: index) -> tensor<10x?xf64, #CSC> {
  %0 = bufferization.alloc_tensor(%arg0) : tensor<10x?xf64, #CSC>
  %1 = sparse_tensor.load %0 : tensor<10x?xf64, #CSC>
  return %1 : tensor<10x?xf64, #CSC>
}

// CHECK-LABEL: func @sparse_alloc_3d() ->
//  CHECK-SAME: memref<3xindex>, memref<1xindex>, memref<?xf64>
//   CHECK-DAG: %[[F0:.*]] = arith.constant 0.000000e+00 : f64
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
//   CHECK-DAG: %[[C20:.*]] = arith.constant 20 : index
//   CHECK-DAG: %[[C30:.*]] = arith.constant 30 : index
//   CHECK-DAG: %[[C6000:.*]] = arith.constant 6000 : index
//       CHECK: %[[A0:.*]] = memref.alloc() : memref<3xindex>
//       CHECK: %[[A1:.*]] = memref.alloc() : memref<1xindex>
//       CHECK: %[[AV:.*]] = memref.alloc() : memref<16xf64>
//       CHECK: %[[A2:.*]] = memref.cast %[[AV]] : memref<16xf64> to memref<?xf64>
//       CHECK: linalg.fill ins(%[[C0]] : index) outs(%[[A1]] : memref<1xindex>)
//       CHECK: memref.store %[[C30]], %[[A0]][%[[C0]]] : memref<3xindex>
//       CHECK: memref.store %[[C10]], %[[A0]][%[[C1]]] : memref<3xindex>
//       CHECK: memref.store %[[C20]], %[[A0]][%[[C2]]] : memref<3xindex>
//       CHECK: %[[P:.*]] = sparse_tensor.push_back %[[A1]], %[[A2]], %[[F0]], %[[C6000]]
//       CHECK: return %[[A0]], %[[A1]], %[[P]] :
//  CHECK-SAME:   memref<3xindex>, memref<1xindex>, memref<?xf64>
func.func @sparse_alloc_3d() -> tensor<10x20x30xf64, #Dense3D> {
  %0 = bufferization.alloc_tensor() : tensor<10x20x30xf64, #Dense3D>
  %1 = sparse_tensor.load %0 : tensor<10x20x30xf64, #Dense3D>
  return %1 : tensor<10x20x30xf64, #Dense3D>
}

// CHECK-LABEL: func.func @sparse_expansion1()
//       CHECK: %[[A:.*]] = memref.alloc() : memref<8xf64>
//       CHECK: %[[B:.*]] = memref.alloc() : memref<8xi1>
//       CHECK: %[[C:.*]] = memref.alloc() : memref<8xindex>
//       CHECK: %[[D:.*]] = memref.cast %[[C]] : memref<8xindex> to memref<?xindex>
//   CHECK-DAG: linalg.fill ins(%{{.*}}  : f64) outs(%[[A]] : memref<8xf64>)
//   CHECK-DAG: linalg.fill ins(%{{.*}}  : i1) outs(%[[B]] : memref<8xi1>)
//       CHECK: return %[[D]] : memref<?xindex>
func.func @sparse_expansion1() -> memref<?xindex> {
  %0 = bufferization.alloc_tensor() : tensor<4x8xf64, #CSR>
  %values, %filled, %added, %count = sparse_tensor.expand %0
    : tensor<4x8xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return %added : memref<?xindex>
}

// CHECK-LABEL: func.func @sparse_expansion2()
//       CHECK: %[[A:.*]] = memref.alloc() : memref<4xf64>
//       CHECK: %[[B:.*]] = memref.alloc() : memref<4xi1>
//       CHECK: %[[C:.*]] = memref.alloc() : memref<4xindex>
//       CHECK: %[[D:.*]] = memref.cast %[[C]] : memref<4xindex> to memref<?xindex>
//   CHECK-DAG: linalg.fill ins(%{{.*}}  : f64) outs(%[[A]] : memref<4xf64>)
//   CHECK-DAG: linalg.fill ins(%{{.*}}  : i1) outs(%[[B]] : memref<4xi1>)
//       CHECK: return %[[D]] : memref<?xindex>
func.func @sparse_expansion2() -> memref<?xindex> {
  %0 = bufferization.alloc_tensor() : tensor<4x8xf64, #CSC>
  %values, %filled, %added, %count = sparse_tensor.expand %0
    : tensor<4x8xf64, #CSC> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return %added : memref<?xindex>
}

// CHECK-LABEL: func.func @sparse_expansion3(
//  CHECK-SAME: %[[D0:.*]]: index,
//  CHECK-SAME: %{{.*}}: index) -> memref<?xindex> {
//       CHECK: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: %[[S0:.*]] = memref.alloc() : memref<2xindex>
//       CHECK: memref.store %[[D0]], %[[S0]]{{\[}}%[[C1]]] : memref<2xindex>
//       CHECK: %[[D1:.*]] = memref.load %[[S0]]{{\[}}%[[C1]]] : memref<2xindex>
//       CHECK: %[[V:.*]] = memref.alloc(%[[D1]]) : memref<?xf64>
//       CHECK: %[[B:.*]] = memref.alloc(%[[D1]]) : memref<?xi1>
//       CHECK: %[[D:.*]] = memref.alloc(%[[D1]]) : memref<?xindex>
//       CHECK: linalg.fill ins(%{{.*}} : f64) outs(%[[V]] : memref<?xf64>)
//       CHECK: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<?xi1>)
//       CHECK: return %[[D]] : memref<?xindex>
func.func @sparse_expansion3(%arg0: index, %arg1: index) -> memref<?xindex> {
  %0 = bufferization.alloc_tensor(%arg0, %arg1) : tensor<?x?xf64, #CSC>
  %values, %filled, %added, %count = sparse_tensor.expand %0
    : tensor<?x?xf64, #CSC> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return %added : memref<?xindex>
}

// CHECK-LABEL: func @sparse_compression_1d(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xindex>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xf64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xi1>,
//  CHECK-SAME: %[[A7:.*7]]: memref<?xindex>,
//  CHECK-SAME: %[[A8:.*8]]: index)
//   CHECK-DAG: %[[B0:.*]] = arith.constant false
//   CHECK-DAG: %[[F0:.*]] = arith.constant 0.000000e+00 : f64
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: sparse_tensor.sort %[[A8]], %[[A7]] : memref<?xindex>
//       CHECK: %[[R:.*]]:2 = scf.for %[[I:.*]] = %[[C0]] to %[[A8]] step %[[C1]] iter_args(%[[P0:.*]] = %[[A3]], %[[P1:.*]] = %[[A4]]) -> (memref<?xindex>, memref<?xf64>) {
//       CHECK:   %[[INDEX:.*]] = memref.load %[[A7]][%[[I]]] : memref<?xindex>
//       CHECK:   %[[VAL:.*]] = memref.load %[[A5]][%[[INDEX]]] : memref<?xf64>
//       CHECK:   %[[PV:.*]] = sparse_tensor.push_back %[[A1]], %[[P1]], %[[VAL]] {idx = 2 : index} : memref<3xindex>, memref<?xf64>, f64
//       CHECK:   memref.store %[[F0]], %[[A5]][%[[INDEX]]] : memref<?xf64>
//       CHECK:   memref.store %[[B0]], %[[A6]][%[[INDEX]]] : memref<?xi1>
//       CHECK:   scf.yield %{{.*}}, %[[PV]] : memref<?xindex>, memref<?xf64>
//       CHECK: }
//       CHECK: memref.dealloc %[[A5]] : memref<?xf64>
//       CHECK: memref.dealloc %[[A6]] : memref<?xi1>
//       CHECK: memref.dealloc %[[A7]] : memref<?xindex>
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[R]]#0, %[[R]]#1
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>
func.func @sparse_compression_1d(%tensor: tensor<100xf64, #SV>,
                                 %values: memref<?xf64>,
                                 %filled: memref<?xi1>,
                                 %added: memref<?xindex>,
                                 %count: index) -> tensor<100xf64, #SV> {
  %0 = sparse_tensor.compress %values, %filled, %added, %count into %tensor[]
    : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<100xf64, #SV>
  %1 = sparse_tensor.load %0 hasInserts : tensor<100xf64, #SV>
  return %1 : tensor<100xf64, #SV>
}

// CHECK-LABEL: func @sparse_compression(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xf64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xi1>,
//  CHECK-SAME: %[[A7:.*7]]: memref<?xindex>,
//  CHECK-SAME: %[[A8:.*8]]: index,
//  CHECK-SAME: %[[A9:.*9]]: index)
//   CHECK-DAG: %[[B0:.*]] = arith.constant false
//   CHECK-DAG: %[[F0:.*]] = arith.constant 0.000000e+00 : f64
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: sparse_tensor.sort %[[A8]], %[[A7]] : memref<?xindex>
//       CHECK: %[[R:.*]]:2 = scf.for %[[I:.*]] = %[[C0]] to %[[A8]] step %[[C1]] iter_args(%[[P0:.*]] = %[[A3]], %[[P1:.*]] = %[[A4]]) -> (memref<?xi64>, memref<?xf64>) {
//       CHECK:   %[[INDEX:.*]] = memref.load %[[A7]][%[[I]]] : memref<?xindex>
//       CHECK:   %[[VAL:.*]] = memref.load %[[A5]][%[[INDEX]]] : memref<?xf64>
//       CHECK:   %[[PV:.*]] = sparse_tensor.push_back %[[A1]], %[[P1]], %[[VAL]] {idx = 2 : index} : memref<3xindex>, memref<?xf64>, f64
//       CHECK:   memref.store %[[F0]], %[[A5]][%[[INDEX]]] : memref<?xf64>
//       CHECK:   memref.store %[[B0]], %[[A6]][%[[INDEX]]] : memref<?xi1>
//       CHECK:   scf.yield %{{.*}}, %[[PV]] : memref<?xi64>, memref<?xf64>
//       CHECK: }
//       CHECK: memref.dealloc %[[A5]] : memref<?xf64>
//       CHECK: memref.dealloc %[[A6]] : memref<?xi1>
//       CHECK: memref.dealloc %[[A7]] : memref<?xindex>
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[R]]#0, %[[R]]#1
//  CHECK-SAME:   memref<2xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>
func.func @sparse_compression(%tensor: tensor<8x8xf64, #CSR>,
                              %values: memref<?xf64>,
                              %filled: memref<?xi1>,
                              %added: memref<?xindex>,
                              %count: index,
                              %i: index) -> tensor<8x8xf64, #CSR> {
  %0 = sparse_tensor.compress %values, %filled, %added, %count into %tensor[%i]
    : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<8x8xf64, #CSR>
  %1 = sparse_tensor.load %0 hasInserts : tensor<8x8xf64, #CSR>
  return %1 : tensor<8x8xf64, #CSR>
}

// CHECK-LABEL: func @sparse_compression_unordered(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xindex>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xf64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xi1>,
//  CHECK-SAME: %[[A7:.*7]]: memref<?xindex>,
//  CHECK-SAME: %[[A8:.*8]]: index,
//  CHECK-SAME: %[[A9:.*9]]: index)
//   CHECK-DAG: %[[B0:.*]] = arith.constant false
//   CHECK-DAG: %[[F0:.*]] = arith.constant 0.000000e+00 : f64
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-NOT: sparse_tensor.sort
//       CHECK: %[[R:.*]]:2 = scf.for %[[I:.*]] = %[[C0]] to %[[A8]] step %[[C1]] iter_args(%[[P0:.*]] = %[[A3]], %[[P1:.*]] = %[[A4]]) -> (memref<?xindex>, memref<?xf64>) {
//       CHECK:   %[[INDEX:.*]] = memref.load %[[A7]][%[[I]]] : memref<?xindex>
//       CHECK:   %[[VAL:.*]] = memref.load %[[A5]][%[[INDEX]]] : memref<?xf64>
//       CHECK:   %[[PV:.*]] = sparse_tensor.push_back %[[A1]], %[[P1]], %[[VAL]] {idx = 2 : index} : memref<3xindex>, memref<?xf64>, f64
//       CHECK:   memref.store %[[F0]], %[[A5]][%[[INDEX]]] : memref<?xf64>
//       CHECK:   memref.store %[[B0]], %[[A6]][%[[INDEX]]] : memref<?xi1>
//       CHECK:   scf.yield %{{.*}}, %[[PV]] : memref<?xindex>, memref<?xf64>
//       CHECK: }
//       CHECK: memref.dealloc %[[A5]] : memref<?xf64>
//       CHECK: memref.dealloc %[[A6]] : memref<?xi1>
//       CHECK: memref.dealloc %[[A7]] : memref<?xindex>
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[R]]#0, %[[R]]#1
//  CHECK-SAME:   memref<2xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>
func.func @sparse_compression_unordered(%tensor: tensor<8x8xf64, #UCSR>,
                                        %values: memref<?xf64>,
                                        %filled: memref<?xi1>,
                                        %added: memref<?xindex>,
                                        %count: index,
                                        %i: index) -> tensor<8x8xf64, #UCSR> {
  %0 = sparse_tensor.compress %values, %filled, %added, %count into %tensor[%i]
    : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<8x8xf64, #UCSR>
  %1 = sparse_tensor.load %0 hasInserts : tensor<8x8xf64, #UCSR>
  return %1 : tensor<8x8xf64, #UCSR>
}

// CHECK-LABEL: func @sparse_insert(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xindex>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: index,
//  CHECK-SAME: %[[A6:.*6]]: f64)
//       CHECK: %[[P:.*]] = sparse_tensor.push_back %[[A1]], %[[A4]], %[[A6]]  {idx = 2 : index} : memref<3xindex>, memref<?xf64>, f64
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %{{.*}}, %[[P]] :
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>
func.func @sparse_insert(%arg0: tensor<128xf64, #SV>, %arg1: index, %arg2: f64) -> tensor<128xf64, #SV> {
  %0 = sparse_tensor.insert %arg2 into %arg0[%arg1] : tensor<128xf64, #SV>
  %1 = sparse_tensor.load %0 hasInserts : tensor<128xf64, #SV>
  return %1 : tensor<128xf64, #SV>
}

// CHECK-LABEL: func @sparse_insert_typed(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: index,
//  CHECK-SAME: %[[A6:.*6]]: f64)
//       CHECK: %[[P:.*]] = sparse_tensor.push_back %[[A1]], %[[A4]], %[[A6]]  {idx = 2 : index} : memref<3xindex>, memref<?xf64>, f64
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %{{.*}}, %[[P]] :
//  CHECK-SAME:   memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>
func.func @sparse_insert_typed(%arg0: tensor<128xf64, #SparseVector>, %arg1: index, %arg2: f64) -> tensor<128xf64, #SparseVector> {
  %0 = sparse_tensor.insert %arg2 into %arg0[%arg1] : tensor<128xf64, #SparseVector>
  %1 = sparse_tensor.load %0 hasInserts : tensor<128xf64, #SparseVector>
  return %1 : tensor<128xf64, #SparseVector>
}

// CHECK-LABEL: func.func @sparse_nop_convert(
//  CHECK-SAME: %[[A0:.*0]]: memref<1xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf32>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]] :
//  CHECK-SAME: memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf32>
func.func @sparse_nop_convert(%arg0: tensor<32xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<32xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}
