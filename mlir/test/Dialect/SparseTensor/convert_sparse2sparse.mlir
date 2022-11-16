// First use with `kViaCOO` for sparse2sparse conversion (the old way).
// RUN: mlir-opt %s --sparse-tensor-conversion="s2s-strategy=1" \
// RUN:    --canonicalize --cse | FileCheck %s -check-prefix=CHECK-COO
//
// Now again with `kAuto` (the new default).
// RUN: mlir-opt %s --sparse-tensor-conversion="s2s-strategy=0" \
// RUN:    --canonicalize --cse | FileCheck %s -check-prefixes=CHECK-AUTO,CHECK

// RUN: mlir-opt %s --sparse-tensor-rewrite="enable-runtime-library=false enable-foreach=false" \
// RUN: --canonicalize --cse | FileCheck %s --check-prefix=CHECK-RWT

#SparseVector64 = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseVector32 = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"],
  pointerBitWidth = 32,
  indexBitWidth = 32
}>

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#SortedWRT3D = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton-nu", "singleton" ]

}>

#TsssPermuted = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (k,i,j)>
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

// CHECK-RWT-LABEL: func.func @sparse_convert(
//  CHECK-RWT-SAME: %[[A:.*]]: tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>)
//  CHECK-RWT-DAG:  %[[C0:.*]] = arith.constant 0 : index
//      CHECK-RWT:  %[[D:.*]] = tensor.dim %[[A]], %[[C0]]
//      CHECK-RWT:  %[[I0:.*]] = sparse_tensor.indices %[[A]] {dimension = 0 : index}
//      CHECK-RWT:  %[[NNZ:.*]] = sparse_tensor.number_of_entries %[[A]]
//      CHECK-RWT:  %[[V:.*]] = sparse_tensor.values %[[A]]
//      CHECK-RWT:  sparse_tensor.sort %[[NNZ]], %[[I0]] jointly %[[V]]
//      CHECK-RWT:  %[[DST:.*]] = bufferization.alloc_tensor(%[[D]])
//      CHECK-RWT:  %[[RET:.*]] = sparse_tensor.foreach in %[[A]] init(%[[DST]])
//      CHECK-RWT:  ^bb0(%[[FI2:.*]]: index, %[[FV2:.*]]: f32, %[[T:.*]]: tensor<?xf32,
//      CHECK-RWT:    %[[I:.*]] = sparse_tensor.insert %[[FV2]] into %[[T]]{{\[}}%[[FI2]]]
//      CHECK-RWT:    sparse_tensor.yield %[[I]]
//      CHECK-RWT:  }
//      CHECK-RWT:  %[[T:.*]] = sparse_tensor.load %[[RET]] hasInserts
//      CHECK-RWT:  %[[R:.*]] = sparse_tensor.convert %[[T]]
//      CHECK-RWT:  return %[[R]] : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 32, indexBitWidth = 32 }>>
func.func @sparse_convert(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

#SparseSingleton64 = #sparse_tensor.encoding<{
  dimLevelType = ["singleton"],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseSingleton32 = #sparse_tensor.encoding<{
  dimLevelType = ["singleton"],
  pointerBitWidth = 32,
  indexBitWidth = 32
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

// CHECK-WRT-LABEL: func.func @sparse_convert_permuted(
//  CHECK-WRT-SAME: %[[COO:.*]]:
//   CHECK-WRT-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-WRT-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-WRT-DAG: %[[C2:.*]] = arith.constant 2 : index
//       CHECK-WRT: %[[D0:.*]] = tensor.dim %[[COO]], %[[C0]]
//       CHECK-WRT: %[[D1:.*]] = tensor.dim %[[COO]], %[[C1]]
//       CHECK-WRT: %[[D2:.*]] = tensor.dim %[[COO]], %[[C2]]
//       CHECK-WRT: %[[I0:.*]] = sparse_tensor.indices %[[COO]] {dimension = 0 : index}
//       CHECK-WRT: %[[I1:.*]] = sparse_tensor.indices %[[COO]] {dimension = 1 : index}
//       CHECK-WRT: %[[I2:.*]] = sparse_tensor.indices %[[COO]] {dimension = 2 : index}
//       CHECK-WRT: %[[NNZ:.*]] = sparse_tensor.number_of_entries %[[COO]]
//       CHECK-WRT: %[[V:.*]] = sparse_tensor.values %[[COO]]
//       CHECK-WRT: sparse_tensor.sort %[[NNZ]], %[[I2]], %[[I0]], %[[I1]] jointly %[[V]]
//       CHECK-WRT: %[[T1:.*]] = bufferization.alloc_tensor(%[[D0]], %[[D1]], %[[D2]])
//       CHECK-WRT: %[[T2:.*]] = sparse_tensor.foreach in %[[COO]] init(%[[T1]])
//       CHECK-WRT: ^bb0(%[[LI0:.*]]: index, %[[LI1:.*]]: index, %[[LI2:.*]]: index, %[[LV:.*]]: f32, %[[LT1:.*]]: tensor<?x?x?xf32,
//       CHECK-WRT:   %[[LT2:.*]] = sparse_tensor.insert %[[LV]] into %[[LT1]]{{\[}}%[[LI2]], %[[LI0]], %[[LI1]]]
//       CHECK-WRT:   sparse_tensor.yield %[[LT2]]
//       CHECK-WRT: }
//       CHECK-WRT: %[[T3:.*]] = sparse_tensor.load %[[T2:.*]] hasInserts
//       CHECK-WRT: %[[T4:.*]] = sparse_tensor.convert %[[T3]]
//       CHECK-WRT: return %[[T4]]
func.func @sparse_convert_permuted(%arg0: tensor<?x?x?xf32, #SortedCOO3D>) -> tensor<?x?x?xf32, #TsssPermuted> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf32, #SortedCOO3D> to tensor<?x?x?xf32, #TsssPermuted>
  return %0 : tensor<?x?x?xf32, #TsssPermuted>
}
