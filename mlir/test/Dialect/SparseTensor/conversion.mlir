// RUN: mlir-opt %s --lower-sparse-ops-to-foreach --lower-sparse-foreach-to-scf --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

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

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : dense, d0 : compressed, d1 : compressed)
}>

// CHECK-LABEL: func @sparse_nop(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr) -> !llvm.ptr
//       CHECK: return %[[A]] : !llvm.ptr
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_dim1d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[D:.*]] = call @sparseLvlSize(%[[A]], %[[C]])
//       CHECK: return %[[D]] : index
func.func @sparse_dim1d(%arg0: tensor<?xf64, #SparseVector>) -> index {
  %c = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c : tensor<?xf64, #SparseVector>
  return %0 : index
}

// Querying the size of dimension 1 should do so; i.e., it should
// not be permuted into a query for the size of level 2 (even though
// dimension 1 is stored as level 2).
// CHECK-LABEL: func @sparse_dim3d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 2 : index
//       CHECK: %[[D:.*]] = call @sparseLvlSize(%[[A]], %[[C]])
//       CHECK: return %[[D]] : index
func.func @sparse_dim3d(%arg0: tensor<?x?x?xf64, #SparseTensor>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<?x?x?xf64, #SparseTensor>
  return %0 : index
}

// Querying the size of a static dimension should be folded into a
// constant (and we should be sure to get the size of dimension 1,
// not dimension 2 nor level 1).
// CHECK-LABEL: func @sparse_dim3d_const(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 20 : index
//       CHECK: return %[[C]] : index
func.func @sparse_dim3d_const(%arg0: tensor<10x20x30xf64, #SparseTensor>) -> index {
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<10x20x30xf64, #SparseTensor>
  return %0 : index
}

// CHECK-LABEL: func @sparse_new1d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr) -> !llvm.ptr
//   CHECK-DAG: %[[DimShape0:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[DimShape:.*]] = memref.cast %[[DimShape0]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[Reader:.*]] = call @createCheckedSparseTensorReader(%[[A]], %[[DimShape]], %{{.*}})
//   CHECK-DAG: %[[LvlTypes0:.*]] = memref.alloca() : memref<1xi64>
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.cast %[[LvlTypes0]] : memref<1xi64> to memref<?xi64>
//   CHECK-DAG: %[[Iota0:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[Iota:.*]] = memref.cast %[[Iota0]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[DimShape]], %[[DimShape]], %[[LvlTypes]], %[[Iota]], %[[Iota]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[Reader]])
//       CHECK: call @delSparseTensorReader(%[[Reader]])
//       CHECK: return %[[T]] : !llvm.ptr
func.func @sparse_new1d(%arg0: !llvm.ptr) -> tensor<128xf64, #SparseVector> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr to tensor<128xf64, #SparseVector>
  return %0 : tensor<128xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_new2d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr) -> !llvm.ptr
//   CHECK-DAG: %[[DimShape0:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[DimShape:.*]] = memref.cast %[[DimShape0]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[Reader:.*]] = call @createCheckedSparseTensorReader(%[[A]], %[[DimShape]], %{{.*}})
//       CHECK: %[[DimSizes:.*]] = call @getSparseTensorReaderDimSizes(%[[Reader]])
//   CHECK-DAG: %[[LvlTypes0:.*]] = memref.alloca() : memref<2xi64>
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.cast %[[LvlTypes0]] : memref<2xi64> to memref<?xi64>
//   CHECK-DAG: %[[Iota0:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[Iota:.*]] = memref.cast %[[Iota0]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[DimSizes]], %[[DimSizes]], %[[LvlTypes]], %[[Iota]], %[[Iota]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[Reader]])
//       CHECK: call @delSparseTensorReader(%[[Reader]])
//       CHECK: return %[[T]] : !llvm.ptr
func.func @sparse_new2d(%arg0: !llvm.ptr) -> tensor<?x?xf32, #CSR> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr to tensor<?x?xf32, #CSR>
  return %0 : tensor<?x?xf32, #CSR>
}

// CHECK-LABEL: func @sparse_new3d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr) -> !llvm.ptr
//   CHECK-DAG: %[[DimShape0:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[DimShape:.*]] = memref.cast %[[DimShape0]] : memref<3xindex> to memref<?xindex>
//       CHECK: %[[Reader:.*]] = call @createCheckedSparseTensorReader(%[[A]], %[[DimShape]], %{{.*}})
//       CHECK: %[[DimSizes:.*]] = call @getSparseTensorReaderDimSizes(%[[Reader]])
//   CHECK-DAG: %[[LvlTypes0:.*]] = memref.alloca() : memref<3xi64>
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.cast %[[LvlTypes0]] : memref<3xi64> to memref<?xi64>
//   CHECK-DAG: %[[Dim2Lvl0:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[Dim2Lvl:.*]] = memref.cast %[[Dim2Lvl0]] : memref<3xindex> to memref<?xindex>
//   CHECK-DAG: %[[Lvl2Dim0:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[Lvl2Dim:.*]] = memref.cast %[[Lvl2Dim0]] : memref<3xindex> to memref<?xindex>
//   CHECK-DAG: %[[LvlSizes0:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[LvlSizes:.*]] = memref.cast %[[LvlSizes0]] : memref<3xindex> to memref<?xindex>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[DimSizes]], %[[LvlSizes]], %[[LvlTypes]], %[[Dim2Lvl]], %[[Lvl2Dim]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[Reader]])
//       CHECK: call @delSparseTensorReader(%[[Reader]])
//       CHECK: return %[[T]] : !llvm.ptr
func.func @sparse_new3d(%arg0: !llvm.ptr) -> tensor<?x?x?xf32, #SparseTensor> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr to tensor<?x?x?xf32, #SparseTensor>
  return %0 : tensor<?x?x?xf32, #SparseTensor>
}

// CHECK-LABEL: func @sparse_init(
//  CHECK-SAME: %[[I:.*]]: index,
//  CHECK-SAME: %[[J:.*]]: index) -> !llvm.ptr
//   CHECK-DAG: %[[Empty:.*]] = arith.constant 0 : i32
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[LvlTypes0:.*]] = memref.alloca() : memref<2xi64>
//   CHECK-DAG: %[[Sizes0:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[Iota0:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.cast %[[LvlTypes0]] : memref<2xi64> to memref<?xi64>
//   CHECK-DAG: %[[Sizes:.*]] = memref.cast %[[Sizes0]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[Iota:.*]] = memref.cast %[[Iota0]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: memref.store %[[I]], %[[Sizes0]][%[[C0]]] : memref<2xindex>
//   CHECK-DAG: memref.store %[[J]], %[[Sizes0]][%[[C1]]] : memref<2xindex>
//   CHECK-DAG: %[[NP:.*]] = llvm.mlir.zero : !llvm.ptr
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[Sizes]], %[[Sizes]], %[[LvlTypes]], %[[Iota]], %[[Iota]], %{{.*}}, %{{.*}}, %{{.*}}, %[[Empty]], %[[NP]])
//       CHECK: return %[[T]] : !llvm.ptr
func.func @sparse_init(%arg0: index, %arg1: index) -> tensor<?x?xf64, #CSR> {
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64, #CSR>
  %1 = sparse_tensor.load %0 : tensor<?x?xf64, #CSR>
  return %1 : tensor<?x?xf64, #CSR>
}

// CHECK-LABEL: func @sparse_release(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: call @delSparseTensor(%[[A]]) : (!llvm.ptr) -> ()
//       CHECK: return
func.func @sparse_release(%arg0: tensor<128xf64, #SparseVector>) {
  bufferization.dealloc_tensor %arg0 : tensor<128xf64, #SparseVector>
  return
}

// CHECK-LABEL: func @sparse_nop_cast(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr) -> !llvm.ptr
//       CHECK: return %[[A]] : !llvm.ptr
func.func @sparse_nop_cast(%arg0: tensor<64xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %arg0 : tensor<64xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_positions(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparsePositions0(%[[A]], %[[C]]) : (!llvm.ptr, index) -> memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func.func @sparse_positions(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %0 = sparse_tensor.positions %arg0 { level = 0 : index } : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// CHECK-LABEL: func @sparse_positions64(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparsePositions64(%[[A]], %[[C]]) : (!llvm.ptr, index) -> memref<?xi64>
//       CHECK: return %[[T]] : memref<?xi64>
func.func @sparse_positions64(%arg0: tensor<128xf64, #SparseVector64>) -> memref<?xi64> {
  %0 = sparse_tensor.positions %arg0 { level = 0 : index } : tensor<128xf64, #SparseVector64> to memref<?xi64>
  return %0 : memref<?xi64>
}

// CHECK-LABEL: func @sparse_positions32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparsePositions32(%[[A]], %[[C]]) : (!llvm.ptr, index) -> memref<?xi32>
//       CHECK: return %[[T]] : memref<?xi32>
func.func @sparse_positions32(%arg0: tensor<128xf64, #SparseVector32>) -> memref<?xi32> {
  %0 = sparse_tensor.positions %arg0 { level = 0 : index } : tensor<128xf64, #SparseVector32> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_indices(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparseCoordinates0(%[[A]], %[[C]]) : (!llvm.ptr, index) -> memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func.func @sparse_indices(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %0 = sparse_tensor.coordinates %arg0 { level = 0 : index } : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// CHECK-LABEL: func @sparse_indices64(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparseCoordinates64(%[[A]], %[[C]]) : (!llvm.ptr, index) -> memref<?xi64>
//       CHECK: return %[[T]] : memref<?xi64>
func.func @sparse_indices64(%arg0: tensor<128xf64, #SparseVector64>) -> memref<?xi64> {
  %0 = sparse_tensor.coordinates %arg0 { level = 0 : index } : tensor<128xf64, #SparseVector64> to memref<?xi64>
  return %0 : memref<?xi64>
}

// CHECK-LABEL: func @sparse_indices32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparseCoordinates32(%[[A]], %[[C]]) : (!llvm.ptr, index) -> memref<?xi32>
//       CHECK: return %[[T]] : memref<?xi32>
func.func @sparse_indices32(%arg0: tensor<128xf64, #SparseVector32>) -> memref<?xi32> {
  %0 = sparse_tensor.coordinates %arg0 { level = 0 : index } : tensor<128xf64, #SparseVector32> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_valuesf64(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[T:.*]] = call @sparseValuesF64(%[[A]]) : (!llvm.ptr) -> memref<?xf64>
//       CHECK: return %[[T]] : memref<?xf64>
func.func @sparse_valuesf64(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<128xf64, #SparseVector> to memref<?xf64>
  return %0 : memref<?xf64>
}

// CHECK-LABEL: func @sparse_valuesf32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[T:.*]] = call @sparseValuesF32(%[[A]]) : (!llvm.ptr) -> memref<?xf32>
//       CHECK: return %[[T]] : memref<?xf32>
func.func @sparse_valuesf32(%arg0: tensor<128xf32, #SparseVector>) -> memref<?xf32> {
  %0 = sparse_tensor.values %arg0: tensor<128xf32, #SparseVector> to memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK-LABEL: func @sparse_valuesi32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[T:.*]] = call @sparseValuesI32(%[[A]]) : (!llvm.ptr) -> memref<?xi32>
//       CHECK: return %[[T]] : memref<?xi32>
func.func @sparse_valuesi32(%arg0: tensor<128xi32, #SparseVector>) -> memref<?xi32> {
  %0 = sparse_tensor.values %arg0: tensor<128xi32, #SparseVector> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_valuesi16(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[T:.*]] = call @sparseValuesI16(%[[A]]) : (!llvm.ptr) -> memref<?xi16>
//       CHECK: return %[[T]] : memref<?xi16>
func.func @sparse_valuesi16(%arg0: tensor<128xi16, #SparseVector>) -> memref<?xi16> {
  %0 = sparse_tensor.values %arg0: tensor<128xi16, #SparseVector> to memref<?xi16>
  return %0 : memref<?xi16>
}

// CHECK-LABEL: func @sparse_valuesi8(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//       CHECK: %[[T:.*]] = call @sparseValuesI8(%[[A]]) : (!llvm.ptr) -> memref<?xi8>
//       CHECK: return %[[T]] : memref<?xi8>
func.func @sparse_valuesi8(%arg0: tensor<128xi8, #SparseVector>) -> memref<?xi8> {
  %0 = sparse_tensor.values %arg0: tensor<128xi8, #SparseVector> to memref<?xi8>
  return %0 : memref<?xi8>
}

// CHECK-LABEL: func @sparse_noe(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr)
//   CHECK-DAG: %[[C:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[T:.*]] = call @sparseValuesF64(%[[A]]) : (!llvm.ptr) -> memref<?xf64>
//       CHECK: %[[NOE:.*]] = memref.dim %[[T]], %[[C]] : memref<?xf64>
//       CHECK: return %[[NOE]] : index
func.func @sparse_noe(%arg0: tensor<128xf64, #SparseVector>) -> index {
  %0 = sparse_tensor.number_of_entries %arg0 : tensor<128xf64, #SparseVector>
  return %0 : index
}

// CHECK-LABEL: func @sparse_reconstruct(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr
//       CHECK: return %[[A]] : !llvm.ptr
func.func @sparse_reconstruct(%arg0: tensor<128xf32, #SparseVector>) -> tensor<128xf32, #SparseVector> {
  %0 = sparse_tensor.load %arg0 : tensor<128xf32, #SparseVector>
  return %0 : tensor<128xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_reconstruct_ins(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr
//       CHECK: call @endLexInsert(%[[A]]) : (!llvm.ptr) -> ()
//       CHECK: return %[[A]] : !llvm.ptr
func.func @sparse_reconstruct_ins(%arg0: tensor<128xf32, #SparseVector>) -> tensor<128xf32, #SparseVector> {
  %0 = sparse_tensor.load %arg0 hasInserts : tensor<128xf32, #SparseVector>
  return %0 : tensor<128xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_insert(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr,
//  CHECK-SAME: %[[B:.*]]: index,
//  CHECK-SAME: %[[C:.*]]: f32) -> !llvm.ptr {
//   CHECK-DAG: %[[M:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[V:.*]] = memref.alloca() : memref<f32>
//   CHECK-DAG: %[[MC:.*]] = memref.cast %[[M]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: memref.store %[[B]], %[[M]][%[[C0]]] : memref<1xindex>
//   CHECK-DAG: memref.store %[[C]], %[[V]][] : memref<f32>
//       CHECK: call @lexInsertF32(%[[A]], %[[MC]], %[[V]]) : (!llvm.ptr, memref<?xindex>, memref<f32>) -> ()
//       CHECK: return %[[A]] : !llvm.ptr
func.func @sparse_insert(%arg0: tensor<128xf32, #SparseVector>,
                         %arg1: index,
                         %arg2: f32) -> tensor<128xf32, #SparseVector> {
  %0 = tensor.insert %arg2 into %arg0[%arg1] : tensor<128xf32, #SparseVector>
  return %0 : tensor<128xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_expansion1()
//       CHECK: %[[N:.*]] = call @newSparseTensor
//       CHECK: %[[A:.*]] = memref.alloc() : memref<8xf64>
//       CHECK: %[[B:.*]] = memref.alloc() : memref<8xi1>
//       CHECK: %[[C:.*]] = memref.alloc() : memref<8xindex>
//       CHECK: %[[D:.*]] = memref.cast %[[C]] : memref<8xindex> to memref<?xindex>
//   CHECK-DAG: linalg.fill ins(%{{.*}} : f64) outs(%[[A]] : memref<8xf64>)
//   CHECK-DAG: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<8xi1>)
//       CHECK: return %[[D]] : memref<?xindex>
func.func @sparse_expansion1() -> memref<?xindex> {
  %0 = tensor.empty() : tensor<4x8xf64, #CSR>
  %values, %filled, %added, %count = sparse_tensor.expand %0
    : tensor<4x8xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return %added : memref<?xindex>
}

// CHECK-LABEL: func @sparse_expansion2()
//       CHECK: %[[N:.*]] = call @newSparseTensor
//       CHECK: %[[A:.*]] = memref.alloc() : memref<4xf64>
//       CHECK: %[[B:.*]] = memref.alloc() : memref<4xi1>
//       CHECK: %[[C:.*]] = memref.alloc() : memref<4xindex>
//       CHECK: %[[D:.*]] = memref.cast %[[C]] : memref<4xindex> to memref<?xindex>
//   CHECK-DAG: linalg.fill ins(%{{.*}} : f64) outs(%[[A]] : memref<4xf64>)
//   CHECK-DAG: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<4xi1>)
//       CHECK: return %[[D]] : memref<?xindex>
func.func @sparse_expansion2() -> memref<?xindex> {
  %0 = tensor.empty() : tensor<4x8xf64, #CSC>
  %values, %filled, %added, %count = sparse_tensor.expand %0
    : tensor<4x8xf64, #CSC> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return %added : memref<?xindex>
}

// CHECK-LABEL: func @sparse_expansion3(
//       CHECK: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: %[[N:.*]] = call @newSparseTensor
//       CHECK: %[[S:.*]] = call @sparseLvlSize(%[[N]], %[[C1]])
//       CHECK: %[[A:.*]] = memref.alloc(%[[S]]) : memref<?xf64>
//       CHECK: %[[B:.*]] = memref.alloc(%[[S]]) : memref<?xi1>
//       CHECK: %[[C:.*]] = memref.alloc(%[[S]]) : memref<?xindex>
//   CHECK-DAG: linalg.fill ins(%{{.*}} : f64) outs(%[[A]] : memref<?xf64>)
//   CHECK-DAG: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<?xi1>)
//       CHECK: return %[[C]] : memref<?xindex>
func.func @sparse_expansion3(%arg0: index, %arg1: index) -> memref<?xindex> {
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64, #CSC>
  %values, %filled, %added, %count = sparse_tensor.expand %0
    : tensor<?x?xf64, #CSC> to memref<?xf64>, memref<?xi1>, memref<?xindex>
  return %added : memref<?xindex>
}

// CHECK-LABEL: func @sparse_compression(
//  CHECK-SAME: %[[A:.*0]]: !llvm.ptr,
//  CHECK-SAME: %[[B:.*1]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*2]]: memref<?xi1>,
//  CHECK-SAME: %[[D:.*3]]: memref<?xindex>,
//  CHECK-SAME: %[[E:.*4]]: index,
//  CHECK-SAME: %[[F:.*5]]: index) -> !llvm.ptr {
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[X:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[X]] : memref<2xindex> to memref<?xindex>
//       CHECK: memref.store %[[F]], %[[X]][%[[C0]]] : memref<2xindex>
//       CHECK: call @expInsertF64(%[[A]], %[[Y]], %[[B]], %[[C]], %[[D]], %[[E]])
//   CHECK-DAG: memref.dealloc %[[B]] : memref<?xf64>
//   CHECK-DAG: memref.dealloc %[[C]] : memref<?xi1>
//   CHECK-DAG: memref.dealloc %[[D]] : memref<?xindex>
//       CHECK: return %[[A]] : !llvm.ptr
func.func @sparse_compression(%tensor: tensor<8x8xf64, #CSR>,
                              %values: memref<?xf64>,
                              %filled: memref<?xi1>,
                              %added: memref<?xindex>,
                              %count: index,
                              %i: index) -> tensor<8x8xf64, #CSR> {
  %0 = sparse_tensor.compress %values, %filled, %added, %count into %tensor[%i]
    : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<8x8xf64, #CSR>
  return %0 : tensor<8x8xf64, #CSR>
}

// CHECK-LABEL: func @sparse_and_dense_init(
//       CHECK: %[[S:.*]] = call @newSparseTensor
//       CHECK: %[[D:.*]] = tensor.empty
//       CHECK: return %[[S]], %[[D]] : !llvm.ptr, tensor<?x?xf64>
func.func @sparse_and_dense_init(%arg0: index, %arg1: index)
           -> (tensor<?x?xf64, #CSR>, tensor<?x?xf64>) {
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64, #CSR>
  %1 = sparse_tensor.load %0 : tensor<?x?xf64, #CSR>
  %2 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64>
  return %1, %2 : tensor<?x?xf64, #CSR>, tensor<?x?xf64>
}
