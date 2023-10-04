// RUN: mlir-opt %s --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

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

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#SortedCOO2D = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton),
}>

#SortedCOO3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique), d2 : singleton)

}>

#TsssPermuted = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : compressed, d0 : compressed, d1 : compressed)
}>

#COOSlice = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(2, 2, 1)>, d1 : #sparse_tensor<slice(12, 13, 1)>) -> (d0 : compressed(nonunique), d1 : singleton)
}>

// CHECK-LABEL:   func.func @sparse_nop_convert(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK:           return %[[VAL_0]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_nop_convert(%arg0: tensor<64xf32, #SparseVector>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32, #SparseVector> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// CHECK-LABEL:   func.func @sparse_hidden_nop_cast(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK:           return %[[VAL_0]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_hidden_nop_cast(%arg0: tensor<32xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<32xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL:   func.func @sparse_convert_1d_ss(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_4]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_6:.*]] = memref.alloca() : memref<1xi8>
// CHECK:           %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<1xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<1xi8>
// CHECK:           %[[VAL_8:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_8]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:           %[[VAL_12:.*]] = call @newSparseTensor(%[[VAL_9]], %[[VAL_9]], %[[VAL_7]], %[[VAL_11]], %[[VAL_11]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_1]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           return %[[VAL_12]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert_1d_ss(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

// CHECK-LABEL:   func.func @sparse_convert(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_4]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_6:.*]] = memref.alloca() : memref<1xi8>
// CHECK:           %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<1xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<1xi8>
// CHECK:           %[[VAL_8:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_8]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:           %[[VAL_12:.*]] = call @newSparseTensor(%[[VAL_9]], %[[VAL_9]], %[[VAL_7]], %[[VAL_11]], %[[VAL_11]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_1]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           return %[[VAL_12]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

#SparseSingleton64 = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed),
  posWidth = 64,
  crdWidth = 64
}>

#SparseSingleton32 = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

//
// CHECK-LABEL:   func.func @sparse_convert_singleton(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 16 : i8
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_4]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_6:.*]] = memref.alloca() : memref<1xi8>
// CHECK:           %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<1xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<1xi8>
// CHECK:           %[[VAL_8:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_8]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:           %[[VAL_12:.*]] = call @newSparseTensor(%[[VAL_9]], %[[VAL_9]], %[[VAL_7]], %[[VAL_11]], %[[VAL_11]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_1]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           return %[[VAL_12]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert_singleton(%arg0: tensor<?xf32, #SparseSingleton64>) -> tensor<?xf32, #SparseSingleton32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseSingleton64> to tensor<?xf32, #SparseSingleton32>
  return %0 : tensor<?xf32, #SparseSingleton32>
}

<<<<<<< HEAD
// CHECK-LABEL:   func.func @sparse_convert_permuted(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_7]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_9:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_10:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_11:.*]] = memref.alloca() : memref<3xi8>
// CHECK:           %[[VAL_12:.*]] = memref.cast %[[VAL_11]] : memref<3xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_11]]{{\[}}%[[VAL_7]]] : memref<3xi8>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_11]]{{\[}}%[[VAL_6]]] : memref<3xi8>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_11]]{{\[}}%[[VAL_5]]] : memref<3xi8>
// CHECK:           %[[VAL_13:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_14:.*]] = memref.cast %[[VAL_13]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_13]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_13]]{{\[}}%[[VAL_6]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_13]]{{\[}}%[[VAL_5]]] : memref<3xindex>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_5]]] : memref<3xindex>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:           %[[VAL_17:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_6]]] : memref<3xindex>
// CHECK:           %[[VAL_18:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_19:.*]] = memref.cast %[[VAL_18]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_18]]{{\[}}%[[VAL_6]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_18]]{{\[}}%[[VAL_5]]] : memref<3xindex>
// CHECK:           %[[VAL_20:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_21:.*]] = memref.cast %[[VAL_20]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_20]]{{\[}}%[[VAL_6]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_20]]{{\[}}%[[VAL_5]]] : memref<3xindex>
// CHECK:           %[[VAL_22:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_23:.*]] = memref.cast %[[VAL_22]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_15]], %[[VAL_22]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_16]], %[[VAL_22]]{{\[}}%[[VAL_6]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_17]], %[[VAL_22]]{{\[}}%[[VAL_5]]] : memref<3xindex>
// CHECK:           %[[VAL_24:.*]] = call @newSparseTensor(%[[VAL_14]], %[[VAL_23]], %[[VAL_12]], %[[VAL_19]], %[[VAL_21]], %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_1]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_25:.*]] = call @newSparseTensor(%[[VAL_14]], %[[VAL_23]], %[[VAL_12]], %[[VAL_19]], %[[VAL_21]], %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_2]], %[[VAL_24]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           call @delSparseTensorCOOF32(%[[VAL_24]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           return %[[VAL_25]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert_permuted(%arg0: tensor<?x?x?xf32, #SortedCOO3D>) -> tensor<?x?x?xf32, #TsssPermuted> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf32, #SortedCOO3D> to tensor<?x?x?xf32, #TsssPermuted>
  return %0 : tensor<?x?x?xf32, #TsssPermuted>
=======
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
//       CHECK-RWT: sparse_tensor.sort  hybrid_quick_sort %[[VAL_7]], %[[VAL_19]] jointly %[[VAL_18]] {ny = 0 : index, perm_map = #map}
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
func.func @sparse_convert_permuted(%arg0: tensor<2x3x4xf32, #SortedCOO3D>) -> tensor<2x3x4xf32, #TsssPermuted> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x3x4xf32, #SortedCOO3D> to tensor<2x3x4xf32, #TsssPermuted>
  return %0 : tensor<2x3x4xf32, #TsssPermuted>
>>>>>>> dbb1ebbabd07 (implement direct convert rewriter (cont.))
}

// CHECK-LABEL:   func.func @sparse_convert_slice(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 13 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 9 : i8
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 16 : i8
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_10]]{{\[}}%[[VAL_5]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<2xi8>
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_12]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_12]]{{\[}}%[[VAL_4]]] : memref<2xindex>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_14]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_14]]{{\[}}%[[VAL_4]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = call @newSparseTensor(%[[VAL_13]], %[[VAL_13]], %[[VAL_11]], %[[VAL_15]], %[[VAL_15]], %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_1]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           return %[[VAL_16]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert_slice(%arg0: tensor<2x13xi32, #COOSlice>) -> (tensor<2x13xi32, #SortedCOO2D>)  {
  %0 = sparse_tensor.convert %arg0 : tensor<2x13xi32, #COOSlice> to tensor<2x13xi32, #SortedCOO2D>
  return %0 : tensor<2x13xi32, #SortedCOO2D>
}
