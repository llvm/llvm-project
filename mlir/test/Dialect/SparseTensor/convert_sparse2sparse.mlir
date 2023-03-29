// First use with `kViaCOO` for sparse2sparse conversion (the old way).
// RUN: mlir-opt %s --sparse-tensor-conversion="s2s-strategy=1" \
// RUN:    --canonicalize --cse | FileCheck %s -check-prefix=CHECK-COO
//
// Now again with `kAuto` (the new default).
// RUN: mlir-opt %s --sparse-tensor-conversion="s2s-strategy=0" \
// RUN:    --canonicalize --cse | FileCheck %s -check-prefixes=CHECK-AUTO,CHECK

// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false enable-foreach=false" \
// RUN: --canonicalize --cse | FileCheck %s --check-prefix=CHECK-RWT

#SparseVector64 = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"],
  posWidth = 64,
  crdWidth = 64
}>

#SparseVector32 = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"],
  posWidth = 32,
  crdWidth = 32
}>

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#SortedCOO2D = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ],
}>

#SortedCOO3D = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton-nu", "singleton" ]

}>

#TsssPermuted = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (k,i,j)>
}>

#COOSlice = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ],
  slice = [ (2, 2, 1), (12, 13, 1) ]
}>

// CHECK-LABEL: func @sparse_nop_convert(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
//       CHECK: return %[[A]] : !llvm.ptr<i8>
func.func @sparse_nop_convert(%arg0: tensor<64xf32, #SparseVector>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32, #SparseVector> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_hidden_nop_cast(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
//       CHECK: return %[[A]] : !llvm.ptr<i8>
func.func @sparse_hidden_nop_cast(%arg0: tensor<32xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<32xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_convert_1d_ss(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//   CHECK-DAG: %[[SparseToSparse:.*]] = arith.constant 3 : i32
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[Iota:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<1xi8> to memref<?xi8>
//   CHECK-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[SparseToSparse]], %[[A]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func.func @sparse_convert_1d_ss(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

// CHECK-COO-LABEL: func @sparse_convert(
//  CHECK-COO-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//   CHECK-COO-DAG: %[[ToCOO:.*]] = arith.constant 5 : i32
//   CHECK-COO-DAG: %[[FromCOO:.*]] = arith.constant 2 : i32
//   CHECK-COO-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-COO-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[Iota:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<1xi8> to memref<?xi8>
//   CHECK-COO-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-COO-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-COO-DAG: %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<1xindex> to memref<?xindex>
//       CHECK-COO: %[[C:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[ToCOO]], %[[A]])
//       CHECK-COO: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[FromCOO]], %[[C]])
//       CHECK-COO: call @delSparseTensorCOOF32(%[[C]])
//       CHECK-COO: return %[[T]] : !llvm.ptr<i8>
//
// CHECK-AUTO-LABEL: func @sparse_convert(
//  CHECK-AUTO-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//   CHECK-AUTO-DAG: %[[SparseToSparse:.*]] = arith.constant 3 : i32
//   CHECK-AUTO-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-AUTO-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[Iota:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<1xi8> to memref<?xi8>
//   CHECK-AUTO-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-AUTO-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-AUTO-DAG: %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<1xindex> to memref<?xindex>
//       CHECK-AUTO: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[SparseToSparse]], %[[A]])
//       CHECK-AUTO: return %[[T]] : !llvm.ptr<i8>

func.func @sparse_convert(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

#SparseSingleton64 = #sparse_tensor.encoding<{
  dimLevelType = ["singleton"],
  posWidth = 64,
  crdWidth = 64
}>

#SparseSingleton32 = #sparse_tensor.encoding<{
  dimLevelType = ["singleton"],
  posWidth = 32,
  crdWidth = 32
}>

// CHECK-COO-LABEL: func @sparse_convert_singleton(
//  CHECK-COO-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//   CHECK-COO-DAG: %[[ToCOO:.*]] = arith.constant 5 : i32
//   CHECK-COO-DAG: %[[FromCOO:.*]] = arith.constant 2 : i32
//   CHECK-COO-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-COO-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[Iota:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<1xi8> to memref<?xi8>
//   CHECK-COO-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-COO-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-COO-DAG: %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<1xindex> to memref<?xindex>
//       CHECK-COO: %[[C:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[ToCOO]], %[[A]])
//       CHECK-COO: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[FromCOO]], %[[C]])
//       CHECK-COO: call @delSparseTensorCOOF32(%[[C]])
//       CHECK-COO: return %[[T]] : !llvm.ptr<i8>
//
// CHECK-AUTO-LABEL: func @sparse_convert_singleton(
//  CHECK-AUTO-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//   CHECK-AUTO-DAG: %[[SparseToSparse:.*]] = arith.constant 3 : i32
//   CHECK-AUTO-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-AUTO-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[Iota:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<1xi8> to memref<?xi8>
//   CHECK-AUTO-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-AUTO-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-AUTO-DAG: %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<1xindex> to memref<?xindex>
//       CHECK-AUTO: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[SparseToSparse]], %[[A]])
//       CHECK-AUTO: return %[[T]] : !llvm.ptr<i8>
func.func @sparse_convert_singleton(%arg0: tensor<?xf32, #SparseSingleton64>) -> tensor<?xf32, #SparseSingleton32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseSingleton64> to tensor<?xf32, #SparseSingleton32>
  return %0 : tensor<?xf32, #SparseSingleton32>
}

// CHECK-RWT-LABEL: func.func @sparse_convert_permuted(
//  CHECK-RWT-SAME: %[[VAL_0:.*]]: tensor<?x?x?xf32, #{{.*}}>>) -> tensor<?x?x?xf32, #{{.*}}>> {
//   CHECK-RWT-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
//   CHECK-RWT-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
//   CHECK-RWT-DAG: %[[VAL_3:.*]] = arith.constant 2 : index
//   CHECK-RWT-DAG: %[[VAL_4:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]]
//   CHECK-RWT-DAG: %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]]
//   CHECK-RWT-DAG: %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]]
//   CHECK-RWT-DAG: %[[VAL_7:.*]] = sparse_tensor.number_of_entries %[[VAL_0]]
//       CHECK-RWT: %[[VAL_8:.*]] = bufferization.alloc_tensor(%[[VAL_4]], %[[VAL_5]], %[[VAL_6]]) size_hint=%[[VAL_7]]
//       CHECK-RWT: %[[VAL_9:.*]] = sparse_tensor.foreach in %[[VAL_0]] init(%[[VAL_8]])
//       CHECK-RWT: ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index, %[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: tensor<?x?x?xf32, #{{.*}}>>):
//       CHECK-RWT:   %[[VAL_15:.*]] = sparse_tensor.insert %[[VAL_13]] into %[[VAL_14]]{{\[}}%[[VAL_12]], %[[VAL_10]], %[[VAL_11]]]
//       CHECK-RWT:   sparse_tensor.yield %[[VAL_15]] : tensor<?x?x?xf32, #{{.*}}>>
//       CHECK-RWT: }
//       CHECK-RWT: %[[VAL_16:.*]] = sparse_tensor.load %[[VAL_17:.*]] hasInserts : tensor<?x?x?xf32, #{{.*}}>>
//       CHECK-RWT: %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_16]] : tensor<?x?x?xf32, #{{.*}}>> to memref<?xf32>
//       CHECK-RWT: %[[VAL_19:.*]] = sparse_tensor.coordinates_buffer %[[VAL_16]] : tensor<?x?x?xf32, #{{.*}}>> to memref<?xindex>
//       CHECK-RWT: sparse_tensor.sort_coo  hybrid_quick_sort %[[VAL_7]], %[[VAL_19]] jointly %[[VAL_18]] {nx = 3 : index, ny = 0 : index}
//       CHECK-RWT: %[[VAL_20:.*]] = bufferization.alloc_tensor(%[[VAL_4]], %[[VAL_5]], %[[VAL_6]]) size_hint=%[[VAL_7]]
//       CHECK-RWT: %[[VAL_21:.*]] = sparse_tensor.foreach in %[[VAL_16]] init(%[[VAL_20]])
//       CHECK-RWT: ^bb0(%[[VAL_22:.*]]: index, %[[VAL_23:.*]]: index, %[[VAL_24:.*]]: index, %[[VAL_25:.*]]: f32, %[[VAL_26:.*]]: tensor<?x?x?xf32, #{{.*}}>>):
//       CHECK-RWT:   %[[VAL_27:.*]] = sparse_tensor.insert %[[VAL_25]] into %[[VAL_26]]{{\[}}%[[VAL_24]], %[[VAL_22]], %[[VAL_23]]]
//       CHECK-RWT:   sparse_tensor.yield %[[VAL_27]]
//       CHECK-RWT: }
//       CHECK-RWT: bufferization.dealloc_tensor %[[VAL_16]]
//       CHECK-RWT: %[[VAL_28:.*]] = sparse_tensor.load %[[VAL_29:.*]] hasInserts
//       CHECK-RWT: %[[VAL_30:.*]] = sparse_tensor.convert %[[VAL_28]]
//       CHECK-RWT: return %[[VAL_30]]
func.func @sparse_convert_permuted(%arg0: tensor<?x?x?xf32, #SortedCOO3D>) -> tensor<?x?x?xf32, #TsssPermuted> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf32, #SortedCOO3D> to tensor<?x?x?xf32, #TsssPermuted>
  return %0 : tensor<?x?x?xf32, #TsssPermuted>
}

// CHECK-RWT-LABEL: func.func @sparse_convert_slice(
//  CHECK-RWT-SAME: %[[VAL_0:.*]]: tensor<2x13xi32, #{{.*}}>) -> tensor<2x13xi32, #{{.*}}> {
//       CHECK-RWT: %[[VAL_1:.*]] = sparse_tensor.number_of_entries %[[VAL_0]] : tensor<2x13xi32, #{{.*}}>
//       CHECK-RWT: %[[VAL_2:.*]] = bufferization.alloc_tensor() size_hint=%[[VAL_1]] : tensor<2x13xi32, #{{.*}}>
//       CHECK-RWT: %[[VAL_3:.*]] = sparse_tensor.foreach in %[[VAL_0]] init(%[[VAL_2]]) : tensor<2x13xi32, #{{.*}}>, tensor<2x13xi32, #{{.*}}> -> tensor<2x13xi32, #{{.*}}> do {
//       CHECK-RWT: ^bb0(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: tensor<2x13xi32, #{{.*}}>):
//       CHECK-RWT:   %[[VAL_8:.*]] = sparse_tensor.insert %[[VAL_6]] into %[[VAL_7]]{{\[}}%[[VAL_4]], %[[VAL_5]]] : tensor<2x13xi32, #{{.*}}>
//       CHECK-RWT:   sparse_tensor.yield %[[VAL_8]] : tensor<2x13xi32, #{{.*}}>
//       CHECK-RWT: }
//       CHECK-RWT: %[[VAL_9:.*]] = sparse_tensor.load %[[VAL_10:.*]] hasInserts : tensor<2x13xi32, #{{.*}}>
//       CHECK-RWT: %[[VAL_11:.*]] = sparse_tensor.convert %[[VAL_9]] : tensor<2x13xi32, #{{.*}}> to tensor<2x13xi32, #{{.*}}>
//       CHECK-RWT: return %[[VAL_11]] : tensor<2x13xi32, #{{.*}}>
func.func @sparse_convert_slice(%arg0: tensor<2x13xi32, #COOSlice>) -> (tensor<2x13xi32, #SortedCOO2D>)  {
  %0 = sparse_tensor.convert %arg0 : tensor<2x13xi32, #COOSlice> to tensor<2x13xi32, #SortedCOO2D>
  return %0 : tensor<2x13xi32, #SortedCOO2D>
}
