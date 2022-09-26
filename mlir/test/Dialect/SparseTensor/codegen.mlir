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
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]] : memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf64>
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
//  CHECK-SAME: %[[A9:.*9]]: memref<?xf64>) ->
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[A8]], %[[A9]]
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
//       CHECK: %[[T0:.*]]:10 = call @sparse_nop_multi_ret(%[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[A8]], %[[A9]])
//       CHECK: return %[[T0]]#0, %[[T0]]#1, %[[T0]]#2, %[[T0]]#3, %[[T0]]#4, %[[T0]]#5, %[[T0]]#6, %[[T0]]#7, %[[T0]]#8, %[[T0]]#9
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
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]] : memref<1xindex>, memref<3xindex>, memref<?xi32>, memref<?xi64>, memref<?xf32>
func.func @sparse_nop_cast(%arg0: tensor<64xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %arg0 : tensor<64xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_cast_3d(
//  CHECK-SAME: %[[A0:.*0]]: memref<3xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<1xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf32>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]] : memref<3xindex>, memref<1xindex>, memref<?xf32>
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
//       CHECK: memref.store %[[A]], %[[T0]][%[[C0]]] : memref<2xindex>
//       CHECK: memref.store %[[C10]], %[[T0]][%[[C1]]] : memref<2xindex>
//       CHECK: %[[T2:.*]] = memref.alloc() : memref<1xindex>
//       CHECK: %[[T3:.*]] = memref.cast %[[T2]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[T4:.*]] = memref.alloc() : memref<1xindex>
//       CHECK: %[[T5:.*]] = memref.cast %[[T4]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[T6:.*]] = memref.alloc() : memref<1xf64>
//       CHECK: %[[T7:.*]] = memref.cast %[[T6]] : memref<1xf64> to memref<?xf64>
//       CHECK: linalg.fill ins(%[[C0]] : index) outs(%[[T1]] : memref<3xindex>)
//       CHECK: return %[[T0]], %[[T1]], %[[T3]], %[[T5]], %[[T7]]
func.func @sparse_alloc_csc(%arg0: index) -> tensor<10x?xf64, #CSC> {
  %0 = bufferization.alloc_tensor(%arg0) : tensor<10x?xf64, #CSC>
  %1 = sparse_tensor.load %0 : tensor<10x?xf64, #CSC>
  return %1 : tensor<10x?xf64, #CSC>
}

// CHECK-LABEL: func @sparse_alloc_3d() ->
//  CHECK-SAME: memref<3xindex>, memref<1xindex>, memref<?xf64>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
//   CHECK-DAG: %[[C20:.*]] = arith.constant 20 : index
//   CHECK-DAG: %[[C30:.*]] = arith.constant 30 : index
//   CHECK-DAG: %[[C6000:.*]] = arith.constant 6000 : index
//       CHECK: %[[A0:.*]] = memref.alloc() : memref<3xindex>
//       CHECK: %[[A1:.*]] = memref.alloc() : memref<1xindex>
//       CHECK: memref.store %[[C30]], %[[A0]][%[[C0]]] : memref<3xindex>
//       CHECK: memref.store %[[C10]], %[[A0]][%[[C1]]] : memref<3xindex>
//       CHECK: memref.store %[[C20]], %[[A0]][%[[C2]]] : memref<3xindex>
//       CHECK: %[[A:.*]] = memref.alloc() : memref<6000xf64>
//       CHECK: %[[A2:.*]] = memref.cast %[[A]] : memref<6000xf64> to memref<?xf64>
//       CHECK: memref.store %[[C6000]], %[[A1]][%[[C0]]] : memref<1xindex>
//       CHECK: return %[[A0]], %[[A1]], %[[A2]] : memref<3xindex>, memref<1xindex>, memref<?xf64>
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
    : tensor<4x8xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
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
    : tensor<4x8xf64, #CSC> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
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
    : tensor<?x?xf64, #CSC> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
  return %added : memref<?xindex>
}

// CHECK-LABEL: func @sparse_compression(
//  CHECK-SAME: %[[A0:.*0]]: memref<2xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<3xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xindex>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xf64>,
//  CHECK-SAME: %[[A7:.*7]]: memref<?xi1>,
//  CHECK-SAME: %[[A8:.*8]]: memref<?xindex>,
//  CHECK-SAME: %[[A9:.*9]]: index)
//   CHECK-DAG: %[[B0:.*]] = arith.constant false
//   CHECK-DAG: %[[F0:.*]] = arith.constant 0.000000e+00 : f64
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//        TODO: sort
//  CHECK-NEXT: scf.for %[[I:.*]] = %[[C0]] to %[[A9]] step %[[C1]] {
//  CHECK-NEXT:   %[[INDEX:.*]] = memref.load %[[A8]][%[[I]]] : memref<?xindex>
//        TODO:   insert
//   CHECK-DAG:   memref.store %[[F0]], %[[A6]][%[[INDEX]]] : memref<?xf64>
//   CHECK-DAG:   memref.store %[[B0]], %[[A7]][%[[INDEX]]] : memref<?xi1>
//  CHECK-NEXT: }
//   CHECK-DAG: memref.dealloc %[[A6]] : memref<?xf64>
//   CHECK-DAG: memref.dealloc %[[A7]] : memref<?xi1>
//   CHECK-DAG: memref.dealloc %[[A8]] : memref<?xindex>
//       CHECK: return
func.func @sparse_compression(%arg0: tensor<8x8xf64, #CSR>,
                              %arg1: memref<?xindex>,
                              %arg2: memref<?xf64>,
                              %arg3: memref<?xi1>,
                              %arg4: memref<?xindex>,
                              %arg5: index) {
  sparse_tensor.compress %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    : tensor<8x8xf64, #CSR>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index
  return
}

// CHECK-LABEL: func @sparse_push_back(
//  CHECK-SAME: %[[A:.*]]: memref<?xindex>,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> memref<?xf64> {
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[S:.*]] = memref.dim %[[B]], %[[C0]]
//       CHECK: %[[P:.*]] = memref.load %[[A]]{{\[}}%[[C2]]]
//       CHECK: %[[T:.*]] = arith.cmpi uge, %[[P]], %[[S]]
//       CHECK: %[[M:.*]] = scf.if %[[T]] -> (memref<?xf64>) {
//       CHECK:  %[[P1:.*]] = arith.muli %[[S]], %[[C2]]
//       CHECK:  %[[M2:.*]] = memref.realloc %[[B]](%[[P1]])
//       CHECK:  scf.yield %[[M2]] : memref<?xf64>
//       CHECK: } else {
//       CHECK:  scf.yield %[[B]] : memref<?xf64>
//       CHECK: }
//       CHECK: memref.store %[[C]], %[[M]]{{\[}}%[[P]]]
//       CHECK: %[[P2:.*]] = arith.addi %[[P]], %[[C1]]
//       CHECK: memref.store %[[P2]], %[[A]]{{\[}}%[[C2]]]
//       CHECK: return %[[M]] : memref<?xf64>
func.func @sparse_push_back(%arg0: memref<?xindex>, %arg1: memref<?xf64>, %arg2: f64) -> memref<?xf64> {
  %0 = sparse_tensor.push_back %arg0, %arg1, %arg2 {idx = 2 : index} : memref<?xindex>, memref<?xf64>, f64 to memref<?xf64>
  return %0 : memref<?xf64>
}
