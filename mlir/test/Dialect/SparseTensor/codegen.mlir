// RUN: mlir-opt %s --sparse-tensor-codegen  --canonicalize -cse | FileCheck %s

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

#Coo = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ]
}>

#ccoo = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed-nu", "singleton" ]
}>

// CHECK-LABEL: func @sparse_nop(
//  CHECK-SAME: %[[A0:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]] :
//  CHECK-SAME:   memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_multi_ret(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi32>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xf64>,
//  CHECK-SAME: %[[A7:.*7]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]] :
//  CHECK-SAME:   memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
//  CHECK-SAME:   memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
func.func @sparse_nop_multi_ret(%arg0: tensor<?xf64, #SparseVector>,
                                %arg1: tensor<?xf64, #SparseVector>) ->
                                (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) {
  return %arg0, %arg1 : tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_call(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A4:.*4]]: memref<?xi32>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi64>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xf64>,
//  CHECK-SAME: %[[A7:.*7]]: !sparse_tensor.storage_specifier
//       CHECK: %[[T:.*]]:8 = call @sparse_nop_multi_ret(%[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]])
//       CHECK: return %[[T]]#0, %[[T]]#1, %[[T]]#2, %[[T]]#3, %[[T]]#4, %[[T]]#5, %[[T]]#6, %[[T]]#7 :
//  CHECK-SAME:   memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
//  CHECK-SAME:   memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
func.func @sparse_nop_call(%arg0: tensor<?xf64, #SparseVector>,
                           %arg1: tensor<?xf64, #SparseVector>) ->
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) {
  %1, %2 = call @sparse_nop_multi_ret(%arg0, %arg1) :
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>) ->
                           (tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
  return %1, %2: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_cast(
//  CHECK-SAME: %[[A0:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xf32>,
//  CHECK-SAME: %[[A3:.*]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]] :
func.func @sparse_nop_cast(%arg0: tensor<64xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %arg0 : tensor<64xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_cast_3d(
//  CHECK-SAME: %[[A0:.*]]: memref<?xf32>,
//  CHECK-SAME: %[[A1:.*]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A0]], %[[A1]] :
//  CHECK-SAME:   memref<?xf32>, !sparse_tensor.storage_specifier
func.func @sparse_nop_cast_3d(%arg0: tensor<10x20x30xf32, #Dense3D>) -> tensor<?x?x?xf32, #Dense3D> {
  %0 = tensor.cast %arg0 : tensor<10x20x30xf32, #Dense3D> to tensor<?x?x?xf32, #Dense3D>
  return %0 : tensor<?x?x?xf32, #Dense3D>
}

// CHECK-LABEL: func @sparse_dense_2d(
//  CHECK-SAME: %[[A0:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A1:.*]]: !sparse_tensor.storage_specifier
//       CHECK: return
func.func @sparse_dense_2d(%arg0: tensor<?x?xf64, #Dense2D>) {
  return
}

// CHECK-LABEL: func @sparse_row(
//  CHECK-SAME: %[[A0:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*]]: !sparse_tensor.storage_specifier
//       CHECK: return
func.func @sparse_row(%arg0: tensor<?x?xf64, #Row>) {
  return
}

// CHECK-LABEL: func @sparse_csr(
//  CHECK-SAME: %[[A0:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*]]: !sparse_tensor.storage_specifier
//       CHECK: return
func.func @sparse_csr(%arg0: tensor<?x?xf64, #CSR>) {
  return
}

// CHECK-LABEL: func @sparse_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: !sparse_tensor.storage_specifier
//       CHECK: return
func.func @sparse_dcsr(%arg0: tensor<?x?xf64, #DCSR>) {
  return
}

//
// Querying for dimension 1 in the tensor type can immediately
// fold using the original static dimension sizes.
//
// CHECK-LABEL: func @sparse_dense_3d(
//  CHECK-SAME: %[[A0:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A1:.*]]: !sparse_tensor.storage_specifier
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
//  CHECK-SAME: %[[A0:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A1:.*]]: !sparse_tensor.storage_specifier
//       CHECK: %[[A2:.*]] = sparse_tensor.storage_specifier.get %[[A1]] dim_sz at 2
//       CHECK: %[[A3:.*]] = arith.index_cast %[[A2]] : i64 to index
//       CHECK: return %[[A3]] : index
func.func @sparse_dense_3d_dyn(%arg0: tensor<?x?x?xf64, #Dense3D>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<?x?x?xf64, #Dense3D>
  return %0 : index
}

// CHECK-LABEL: func @sparse_pointers_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A2]] : memref<?xi32>
func.func @sparse_pointers_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi32> {
  %0 = sparse_tensor.pointers %arg0 { dimension = 1 : index } : tensor<?x?xf64, #DCSR> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_indices_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A3]] : memref<?xi64>
func.func @sparse_indices_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xi64> {
  %0 = sparse_tensor.indices %arg0 { dimension = 1 : index } : tensor<?x?xf64, #DCSR> to memref<?xi64>
  return %0 : memref<?xi64>
}

// CHECK-LABEL: func @sparse_values_dcsr(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xi32>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xi64>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A4]] : memref<?xf64>
func.func @sparse_values_dcsr(%arg0: tensor<?x?xf64, #DCSR>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<?x?xf64, #DCSR> to memref<?xf64>
  return %0 : memref<?xf64>
}

// CHECK-LABEL: func.func @sparse_values_coo(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xindex>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A4]] : memref<?xf64>
func.func @sparse_values_coo(%arg0: tensor<?x?x?xf64, #ccoo>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<?x?x?xf64, #ccoo> to memref<?xf64>
  return %0 : memref<?xf64>
}


// CHECK-LABEL:   func.func @sparse_indices_coo(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xindex>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: !sparse_tensor.storage_specifier
//       CHECK: %[[C2:.*]] = arith.constant 2 : index
//       CHECK: %[[S0:.*]] = sparse_tensor.storage_specifier.get %[[A5]]  idx_mem_sz at 1
//       CHECK: %[[S1:.*]] = arith.index_cast %[[S0]]
//       CHECK: %[[S2:.*]] = arith.divui %[[S1]], %[[C2]] : index
//       CHECK: %[[R1:.*]] = memref.subview %[[A3]][0] {{\[}}%[[S2]]] [2] : memref<?xindex> to memref<?xindex, strided<[2]>>
//       CHECK: %[[R2:.*]] = memref.cast %[[R1]] : memref<?xindex, strided<[2]>> to memref<?xindex, strided<[?], offset: ?>>
//       CHECK: return %[[R2]] : memref<?xindex, strided<[?], offset: ?>>
func.func @sparse_indices_coo(%arg0: tensor<?x?x?xf64, #ccoo>) -> memref<?xindex, strided<[?], offset: ?>> {
  %0 = sparse_tensor.indices  %arg0 { dimension = 1 : index } : tensor<?x?x?xf64, #ccoo> to memref<?xindex, strided<[?], offset: ?>>
  return %0 : memref<?xindex, strided<[?], offset: ?>>
}

// CHECK-LABEL:   func.func @sparse_indices_buffer_coo(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*3]]: memref<?xindex>,
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: !sparse_tensor.storage_specifier
//       CHECK: return %[[A3]] : memref<?xindex>
func.func @sparse_indices_buffer_coo(%arg0: tensor<?x?x?xf64, #ccoo>) -> memref<?xindex> {
  %0 = sparse_tensor.indices_buffer  %arg0 : tensor<?x?x?xf64, #ccoo> to memref<?xindex>
  return %0 : memref<?xindex>
}

// CHECK-LABEL: func @sparse_noe(
//  CHECK-SAME: %[[A0:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*]]: !sparse_tensor.storage_specifier
//       CHECK: %[[A4:.*]] = sparse_tensor.storage_specifier.get %[[A3]] val_mem_sz
//       CHECK: %[[NOE:.*]] = arith.index_cast %[[A4]] : i64 to index
//       CHECK: return %[[NOE]] : index
func.func @sparse_noe(%arg0: tensor<128xf64, #SparseVector>) -> index {
  %0 = sparse_tensor.number_of_entries %arg0 : tensor<128xf64, #SparseVector>
  return %0 : index
}

// CHECK-LABEL: func @sparse_dealloc_csr(
//  CHECK-SAME: %[[A0:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A1:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*]]: !sparse_tensor.storage_specifier
//       CHECK: memref.dealloc %[[A0]] : memref<?xi32>
//       CHECK: memref.dealloc %[[A1]] : memref<?xi64>
//       CHECK: memref.dealloc %[[A2]] : memref<?xf64>
//       CHECK: return
func.func @sparse_dealloc_csr(%arg0: tensor<?x?xf64, #CSR>) {
  bufferization.dealloc_tensor %arg0 : tensor<?x?xf64, #CSR>
  return
}

// CHECK-LABEL:   func.func @sparse_alloc_csc(
//  CHECK-SAME:     %[[A0:.*]]: index) -> (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:     %[[A1:.*]] = arith.constant 10 : i64
//       CHECK:     %[[A2:.*]] = arith.constant 0 : index
//       CHECK:     %[[A3:.*]] = memref.alloc() : memref<16xindex>
//       CHECK:     %[[A4:.*]] = memref.cast %[[A3]] : memref<16xindex> to memref<?xindex>
//       CHECK:     %[[A5:.*]] = memref.alloc() : memref<16xindex>
//       CHECK:     %[[A6:.*]] = memref.cast %[[A5]] : memref<16xindex> to memref<?xindex>
//       CHECK:     %[[A7:.*]] = memref.alloc() : memref<16xf64>
//       CHECK:     %[[A8:.*]] = memref.cast %[[A7]] : memref<16xf64> to memref<?xf64>
//       CHECK:     %[[A9:.*]] = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier
//       CHECK:     %[[A10:.*]] = arith.index_cast %[[A0]] : index to i64
//       CHECK:     %[[A11:.*]] = sparse_tensor.storage_specifier.set %[[A9]]  dim_sz at 0 with %[[A10]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     %[[A12:.*]] = sparse_tensor.storage_specifier.set %[[A11]]  dim_sz at 1 with %[[A1]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     %[[A13:.*]] = sparse_tensor.storage_specifier.get %[[A12]]  ptr_mem_sz at 1 : !sparse_tensor.storage_specifier
//       CHECK:     %[[A14:.*]] = arith.index_cast %[[A13]] : i64 to index
//       CHECK:     %[[A15:.*]], %[[A16:.*]] = sparse_tensor.push_back %[[A14]], %[[A4]], %[[A2]] : index, memref<?xindex>, index
//       CHECK:     %[[A17:.*]] = arith.index_cast %[[A16]] : index to i64
//       CHECK:     %[[A18:.*]] = sparse_tensor.storage_specifier.set %[[A12]]  ptr_mem_sz at 1 with %[[A17]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     %[[A23:.*]], %[[A24:.*]] = sparse_tensor.push_back %[[A16]], %[[A15]], %[[A2]], %[[A0]] : index, memref<?xindex>, index, index
//       CHECK:     %[[A25:.*]] = arith.index_cast %[[A24]] : index to i64
//       CHECK:     %[[A26:.*]] = sparse_tensor.storage_specifier.set %[[A18]]  ptr_mem_sz at 1 with %[[A25]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     return %[[A23]], %[[A6]], %[[A8]], %[[A26]] : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
func.func @sparse_alloc_csc(%arg0: index) -> tensor<10x?xf64, #CSC> {
  %0 = bufferization.alloc_tensor(%arg0) : tensor<10x?xf64, #CSC>
  %1 = sparse_tensor.load %0 : tensor<10x?xf64, #CSC>
  return %1 : tensor<10x?xf64, #CSC>
}

// CHECK-LABEL:   func.func @sparse_alloc_3d() -> (memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:     %[[A0:.*]] = arith.constant 6000 : index
//       CHECK:     %[[A1:.*]] = arith.constant 20 : i64
//       CHECK:     %[[A2:.*]] = arith.constant 10 : i64
//       CHECK:     %[[A3:.*]] = arith.constant 30 : i64
//       CHECK:     %[[A4:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:     %[[A5:.*]] = memref.alloc() : memref<6000xf64>
//       CHECK:     %[[A6:.*]] = memref.cast %[[A5]] : memref<6000xf64> to memref<?xf64>
//       CHECK:     %[[A7:.*]] = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier
//       CHECK:     %[[A8:.*]] = sparse_tensor.storage_specifier.set %[[A7]]  dim_sz at 0 with %[[A3]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     %[[A9:.*]] = sparse_tensor.storage_specifier.set %[[A8]]  dim_sz at 1 with %[[A2]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     %[[A10:.*]] = sparse_tensor.storage_specifier.set %[[A9]]  dim_sz at 2 with %[[A1]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     %[[A11:.*]] = sparse_tensor.storage_specifier.get %[[A10]]  val_mem_sz : !sparse_tensor.storage_specifier
//       CHECK:     %[[A12:.*]] = arith.index_cast %[[A11]] : i64 to index
//       CHECK:     %[[A13:.*]], %[[A14:.*]] = sparse_tensor.push_back %[[A12]], %[[A6]], %[[A4]], %[[A0]] : index, memref<?xf64>, f64, index
//       CHECK:     %[[A15:.*]] = arith.index_cast %[[A14]] : index to i64
//       CHECK:     %[[A16:.*]] = sparse_tensor.storage_specifier.set %[[A10]]  val_mem_sz with %[[A15]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     return %[[A13]], %[[A16]] : memref<?xf64>, !sparse_tensor.storage_specifier
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
//       CHECK: %[[V:.*]] = memref.alloc(%[[D0]]) : memref<?xf64>
//       CHECK: %[[B:.*]] = memref.alloc(%[[D0]]) : memref<?xi1>
//       CHECK: %[[D:.*]] = memref.alloc(%[[D0]]) : memref<?xindex>
//       CHECK: linalg.fill ins(%{{.*}} : f64) outs(%[[V]] : memref<?xf64>)
//       CHECK: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<?xi1>)
//       CHECK: return %[[D]] : memref<?xindex>
func.func @sparse_expansion3(%arg0: index, %arg1: index) -> memref<?xindex> {
  %0 = bufferization.alloc_tensor(%arg0, %arg1) : tensor<?x?xf64, #CSC>
  %values, %filled, %added, %count = sparse_tensor.expand %0
    : tensor<?x?xf64, #CSC> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return %added : memref<?xindex>
}

// CHECK-LABEL: func.func private @_insert_compressed_100_f64_0_0(
//  CHECK-SAME: %[[A1:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A5:.*4]]: index,
//  CHECK-SAME: %[[A6:.*5]]: f64)
//
// CHECK-LABEL: func.func @sparse_compression_1d(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME: %[[A5:.*5]]: memref<?xi1>,
//  CHECK-SAME: %[[A6:.*6]]: memref<?xindex>,
//  CHECK-SAME: %[[A7:.*7]]: index) -> (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
//   CHECK-DAG: %[[A8:.*]] = arith.constant false
//   CHECK-DAG: %[[A9:.*]] = arith.constant 0.000000e+00 : f64
//   CHECK-DAG: %[[A10:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[A11:.*]] = arith.constant 0 : index
//       CHECK: sparse_tensor.sort %[[A7]], %[[A6]] : memref<?xindex>
//       CHECK: %[[A12:.*]]:4 = scf.for %[[A13:.*]] = %[[A11]] to %[[A7]] step %[[A10]] iter_args(%[[A14:.*]] = %[[A0]], %[[A15:.*]] = %[[A1]], %[[A16:.*]] = %[[A2]], %[[A17:.*]] = %[[A3]])
//       CHECK:   %[[A18:.*]] = memref.load %[[A6]]{{\[}}%[[A13]]] : memref<?xindex>
//       CHECK:   %[[A19:.*]] = memref.load %[[A4]]{{\[}}%[[A18]]] : memref<?xf64>
//       CHECK:   %[[A20:.*]]:4 = func.call @_insert_compressed_100_f64_0_0(%[[A14]], %[[A15]], %[[A16]], %[[A17]], %[[A18]], %[[A19]])
//       CHECK:   memref.store %[[A9]], %[[A4]]{{\[}}%[[A18]]] : memref<?xf64>
//       CHECK:   memref.store %[[A8]], %[[A5]]{{\[}}%[[A18]]] : memref<?xi1>
//       CHECK:   scf.yield %[[A20]]#0, %[[A20]]#1, %[[A20]]#2, %[[A20]]#3
//       CHECK: }
//       CHECK: memref.dealloc %[[A4]] : memref<?xf64>
//       CHECK: memref.dealloc %[[A5]] : memref<?xi1>
//       CHECK: memref.dealloc %[[A6]] : memref<?xindex>
//       CHECK: return %[[A21:.*]]#0, %[[A21]]#1, %[[A21]]#2, %[[A21]]#3 : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
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

// CHECK-LABEL: func.func private @_insert_dense_compressed_8_8_f64_64_32(
//  CHECK-SAME: %[[A1:.*0]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*1]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A5:.*4]]: index,
//  CHECK-SAME: %[[A6:.*5]]: index,
//  CHECK-SAME: %[[A7:.*6]]: f64)
//
// CHECK-LABEL:   func.func @sparse_compression(
//  CHECK-SAME:     %[[A0:.*0]]: memref<?xi32>,
//  CHECK-SAME:     %[[A1:.*1]]: memref<?xi64>,
//  CHECK-SAME:     %[[A2:.*2]]: memref<?xf64>,
//  CHECK-SAME:     %[[A3:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME:     %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME:     %[[A5:.*5]]: memref<?xi1>,
//  CHECK-SAME:     %[[A6:.*6]]: memref<?xindex>,
//  CHECK-SAME:     %[[A7:.*7]]: index,
//  CHECK-SAME:     %[[A8:.*8]]: index) -> (memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:     %[[A9:.*]] = arith.constant 0 : i32
//       CHECK:     %[[A10:.*]] = arith.constant false
//       CHECK:     %[[A11:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:     %[[A12:.*]] = arith.constant 1 : index
//       CHECK:     %[[A13:.*]] = arith.constant 0 : index
//       CHECK:     sparse_tensor.sort %[[A7]], %[[A6]] : memref<?xindex>
//       CHECK:     %[[A14:.*]]:4 = scf.for %[[A15:.*]] = %[[A13]] to %[[A7]] step %[[A12]] iter_args(%[[A16:.*]] = %[[A0]], %[[A17:.*]] = %[[A1]], %[[A18:.*]] = %[[A2]], %[[A19:.*]] = %[[A3]]) -> (memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:       %[[A20:.*]] = memref.load %[[A6]]{{\[}}%[[A15]]] : memref<?xindex>
//       CHECK:       %[[A21:.*]] = memref.load %[[A4]]{{\[}}%[[A20]]] : memref<?xf64>
//       CHECK:       %[[A22:.*]]:4 = func.call @_insert_dense_compressed_8_8_f64_64_32(%[[A16]], %[[A17]], %[[A18]], %[[A19]], %[[A8]], %[[A20]], %[[A21]]) : (memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:       memref.store %[[A11]], %[[A4]]{{\[}}%[[A20]]] : memref<?xf64>
//       CHECK:       memref.store %[[A10]], %[[A5]]{{\[}}%[[A20]]] : memref<?xi1>
//       CHECK:       scf.yield %[[A22]]#0, %[[A22]]#1, %[[A22]]#2, %[[A22]]#3 : memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:     }
//       CHECK:     memref.dealloc %[[A4]] : memref<?xf64>
//       CHECK:     memref.dealloc %[[A5]] : memref<?xi1>
//       CHECK:     memref.dealloc %[[A6]] : memref<?xindex>
//       CHECK:     %[[A23:.*]] = sparse_tensor.storage_specifier.get %[[A24:.*]]#3  ptr_mem_sz at 1 : !sparse_tensor.storage_specifier
//       CHECK:     %[[A25:.*]] = arith.index_cast %[[A23]] : i64 to index
//       CHECK:     %[[A26:.*]] = memref.load %[[A24]]#0{{\[}}%[[A13]]] : memref<?xi32>
//       CHECK:     %[[A27:.*]] = scf.for %[[A28:.*]] = %[[A12]] to %[[A25]] step %[[A12]] iter_args(%[[A29:.*]] = %[[A26]]) -> (i32) {
//       CHECK:       %[[A30:.*]] = memref.load %[[A24]]#0{{\[}}%[[A28]]] : memref<?xi32>
//       CHECK:       %[[A31:.*]] = arith.cmpi eq, %[[A30]], %[[A9]] : i32
//       CHECK:       %[[A32:.*]] = arith.select %[[A31]], %[[A29]], %[[A30]] : i32
//       CHECK:       scf.if %[[A31]] {
//       CHECK:         memref.store %[[A29]], %[[A24]]#0{{\[}}%[[A28]]] : memref<?xi32>
//       CHECK:       }
//       CHECK:       scf.yield %[[A32]] : i32
//       CHECK:     }
//       CHECK:     return %[[A24]]#0, %[[A24]]#1, %[[A24]]#2, %[[A24]]#3 : memref<?xi32>, memref<?xi64>, memref<?xf64>, !sparse_tensor.storage_specifier
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

// CHECK-LABEL: func.func private @"_insert_dense_compressed-no_8_8_f64_0_0"(
//  CHECK-SAME: %[[A1:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A5:.*4]]: index,
//  CHECK-SAME: %[[A6:.*5]]: index,
//  CHECK-SAME: %[[A7:.*6]]: f64)
//
// CHECK-LABEL:   func.func @sparse_compression_unordered(
//  CHECK-SAME:     %[[A0:.*0]]: memref<?xindex>,
//  CHECK-SAME:     %[[A1:.*1]]: memref<?xindex>,
//  CHECK-SAME:     %[[A2:.*2]]: memref<?xf64>,
//  CHECK-SAME:     %[[A3:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME:     %[[A4:.*4]]: memref<?xf64>,
//  CHECK-SAME:     %[[A5:.*5]]: memref<?xi1>,
//  CHECK-SAME:     %[[A6:.*6]]: memref<?xindex>,
//  CHECK-SAME:     %[[A7:.*7]]: index,
//  CHECK-SAME:     %[[A8:.*8]]: index) -> (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:     %[[A9:.*]] = arith.constant false
//       CHECK:     %[[A10:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:     %[[A11:.*]] = arith.constant 0 : index
//       CHECK:     %[[A12:.*]] = arith.constant 1 : index
//       CHECK:     %[[A13:.*]]:4 = scf.for %[[A14:.*]] = %[[A11]] to %[[A7]] step %[[A12]] iter_args(%[[A15:.*]] = %[[A0]], %[[A16:.*]] = %[[A1]], %[[A17:.*]] = %[[A2]], %[[A18:.*]] = %[[A3]]) -> (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:       %[[A19:.*]] = memref.load %[[A6]]{{\[}}%[[A14]]] : memref<?xindex>
//       CHECK:       %[[A20:.*]] = memref.load %[[A4]]{{\[}}%[[A19]]] : memref<?xf64>
//       CHECK:       %[[A21:.*]]:4 = func.call @"_insert_dense_compressed-no_8_8_f64_0_0"(%[[A15]], %[[A16]], %[[A17]], %[[A18]], %[[A8]], %[[A19]], %[[A20]]) : (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:       memref.store %[[A10]], %[[A4]]{{\[}}%[[A19]]] : memref<?xf64>
//       CHECK:       memref.store %[[A9]], %[[A5]]{{\[}}%[[A19]]] : memref<?xi1>
//       CHECK:       scf.yield %[[A21]]#0, %[[A21]]#1, %[[A21]]#2, %[[A21]]#3 : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:     }
//       CHECK:     memref.dealloc %[[A4]] : memref<?xf64>
//       CHECK:     memref.dealloc %[[A5]] : memref<?xi1>
//       CHECK:     memref.dealloc %[[A6]] : memref<?xindex>
//       CHECK:     %[[A22:.*]] = sparse_tensor.storage_specifier.get %[[A23:.*]]#3  ptr_mem_sz at 1 : !sparse_tensor.storage_specifier
//       CHECK:     %[[A24:.*]] = arith.index_cast %[[A22]] : i64 to index
//       CHECK:     %[[A25:.*]] = memref.load %[[A23]]#0{{\[}}%[[A11]]] : memref<?xindex>
//       CHECK:     %[[A26:.*]] = scf.for %[[A27:.*]] = %[[A12]] to %[[A24]] step %[[A12]] iter_args(%[[A28:.*]] = %[[A25]]) -> (index) {
//       CHECK:       %[[A29:.*]] = memref.load %[[A23]]#0{{\[}}%[[A27]]] : memref<?xindex>
//       CHECK:       %[[A30:.*]] = arith.cmpi eq, %[[A29]], %[[A11]] : index
//       CHECK:       %[[A31:.*]] = arith.select %[[A30]], %[[A28]], %[[A29]] : index
//       CHECK:       scf.if %[[A30]] {
//       CHECK:         memref.store %[[A28]], %[[A23]]#0{{\[}}%[[A27]]] : memref<?xindex>
//       CHECK:       }
//       CHECK:       scf.yield %[[A31]] : index
//       CHECK:     }
//       CHECK:     return %[[A23]]#0, %[[A23]]#1, %[[A23]]#2, %[[A23]]#3 : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
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

// CHECK-LABEL: func.func private @_insert_compressed_128_f64_0_0(
//  CHECK-SAME: %[[A1:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A5:.*4]]: index,
//  CHECK-SAME: %[[A6:.*5]]: f64)
//
// CHECK-LABEL: func @sparse_insert(
//  CHECK-SAME: %[[A1:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A5:.*4]]: index,
//  CHECK-SAME: %[[A6:.*5]]: f64)
//       CHECK: %[[R:.*]]:4 = call @_insert_compressed_128_f64_0_0(%[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]])
//       CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2, %[[R]]#3
func.func @sparse_insert(%arg0: tensor<128xf64, #SV>, %arg1: index, %arg2: f64) -> tensor<128xf64, #SV> {
  %0 = sparse_tensor.insert %arg2 into %arg0[%arg1] : tensor<128xf64, #SV>
  %1 = sparse_tensor.load %0 hasInserts : tensor<128xf64, #SV>
  return %1 : tensor<128xf64, #SV>
}

// CHECK-LABEL: func.func private @_insert_compressed_128_f64_64_32(
//  CHECK-SAME: %[[A1:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A5:.*]]: index,
//  CHECK-SAME: %[[A6:.*]]: f64)
//
// CHECK-LABEL: func @sparse_insert_typed(
//  CHECK-SAME: %[[A1:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A5:.*]]: index,
//  CHECK-SAME: %[[A6:.*]]: f64)
//       CHECK: %[[R:.*]]:4 = call @_insert_compressed_128_f64_64_32(%[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]])
//       CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2, %[[R]]#3
func.func @sparse_insert_typed(%arg0: tensor<128xf64, #SparseVector>, %arg1: index, %arg2: f64) -> tensor<128xf64, #SparseVector> {
  %0 = sparse_tensor.insert %arg2 into %arg0[%arg1] : tensor<128xf64, #SparseVector>
  %1 = sparse_tensor.load %0 hasInserts : tensor<128xf64, #SparseVector>
  return %1 : tensor<128xf64, #SparseVector>
}

// CHECK-LABEL: func.func private @"_insert_compressed-nu_singleton_5_6_f64_0_0"(
//  CHECK-SAME: %[[A1:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A3:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A4:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A5:.*4]]: index,
//  CHECK-SAME: %[[A5:.*5]]: index,
//  CHECK-SAME: %[[A7:.*6]]: f64)
//
// CHECK-LABEL: func.func @sparse_insert_coo(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*3]]: !sparse_tensor.storage_specifier
//  CHECK-SAME: %[[A4:.*4]]: index,
//  CHECK-SAME: %[[A5:.*5]]: f64)
//       CHECK: %[[R:.*]]:4 = call @"_insert_compressed-nu_singleton_5_6_f64_0_0"(%[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A4]], %[[A5]])
//       CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2, %[[R]]#3
func.func @sparse_insert_coo(%arg0: tensor<5x6xf64, #Coo>, %arg1: index, %arg2: f64) -> tensor<5x6xf64, #Coo> {
  %0 = sparse_tensor.insert %arg2 into %arg0[%arg1, %arg1] : tensor<5x6xf64, #Coo>
  %1 = sparse_tensor.load %0 hasInserts : tensor<5x6xf64, #Coo>
  return %1 : tensor<5x6xf64, #Coo>
}

// CHECK-LABEL: func.func @sparse_nop_convert(
//  CHECK-SAME: %[[A1:.*]]: memref<?xi32>,
//  CHECK-SAME: %[[A2:.*]]: memref<?xi64>,
//  CHECK-SAME: %[[A3:.*]]: memref<?xf32>,
//  CHECK-SAME: %[[A4:.*]]: !sparse_tensor.storage_specifier
//       CHECK: return  %[[A1]], %[[A2]], %[[A3]], %[[A4]] :
//  CHECK-SAME:   memref<?xi32>, memref<?xi64>, memref<?xf32>, !sparse_tensor.storage_specifier
func.func @sparse_nop_convert(%arg0: tensor<32xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<32xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}
