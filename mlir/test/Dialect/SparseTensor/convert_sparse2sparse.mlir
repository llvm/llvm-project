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
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<1xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %[[SparseToSparse]], %[[A]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func.func @sparse_convert_1d_ss(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

// CHECK-COO-LABEL: func @sparse_convert(
//  CHECK-COO-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//  CHECK-COO-DAG:  %[[ToCOO:.*]] = arith.constant 5 : i32
//  CHECK-COO-DAG:  %[[FromCOO:.*]] = arith.constant 2 : i32
//   CHECK-COO-DAG: %[[P:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-COO-DAG: %[[Q:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[R:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<1xi8> to memref<?xi8>
//   CHECK-COO-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<1xindex> to memref<?xindex>
//   CHECK-COO-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<1xindex> to memref<?xindex>
//       CHECK-COO: %[[C:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %[[ToCOO]], %[[A]])
//       CHECK-COO: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %[[FromCOO]], %[[C]])
//       CHECK-COO: call @delSparseTensorCOOF32(%[[C]])
//       CHECK-COO: return %[[T]] : !llvm.ptr<i8>
// CHECK-AUTO-LABEL: func @sparse_convert(
//  CHECK-AUTO-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//   CHECK-AUTO-DAG: %[[SparseToSparse:.*]] = arith.constant 3 : i32
//   CHECK-AUTO-DAG: %[[P:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-AUTO-DAG: %[[Q:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[R:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<1xi8> to memref<?xi8>
//   CHECK-AUTO-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<1xindex> to memref<?xindex>
//   CHECK-AUTO-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<1xindex> to memref<?xindex>
//       CHECK-AUTO: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %[[SparseToSparse]], %[[A]])
//       CHECK-AUTO: return %[[T]] : !llvm.ptr<i8>

// CHECK-RWT-LABEL: func.func @sparse_convert(
//  CHECK-RWT-SAME: %[[A:.*]]: tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>)
//  CHECK-RWT-DAG:  %[[C0:.*]] = arith.constant 0 : index
//  CHECK-RWT-DAG:  %[[C1:.*]] = arith.constant 1 : index
//      CHECK-RWT:  %[[D:.*]] = tensor.dim %[[A]], %[[C0]]
//      CHECK-RWT:  %[[I0:.*]] = sparse_tensor.indices %[[A]] {dimension = 0 : index}
//      CHECK-RWT:  %[[NNZr:.*]] = memref.load %[[I0]]{{\[}}%[[C1]]] : memref<?xi64>
//      CHECK-RWT:  %[[NNZ:.*]] = arith.index_cast %[[NNZr]] : i64 to index
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
//  CHECK-COO-DAG:  %[[ToCOO:.*]] = arith.constant 5 : i32
//  CHECK-COO-DAG:  %[[FromCOO:.*]] = arith.constant 2 : i32
//   CHECK-COO-DAG: %[[P:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-COO-DAG: %[[Q:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[R:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-COO-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<1xi8> to memref<?xi8>
//   CHECK-COO-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<1xindex> to memref<?xindex>
//   CHECK-COO-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<1xindex> to memref<?xindex>
//       CHECK-COO: %[[C:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %[[ToCOO]], %[[A]])
//       CHECK-COO: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %[[FromCOO]], %[[C]])
//       CHECK-COO: call @delSparseTensorCOOF32(%[[C]])
//       CHECK-COO: return %[[T]] : !llvm.ptr<i8>
// CHECK-AUTO-LABEL: func @sparse_convert_singleton(
//  CHECK-AUTO-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//   CHECK-AUTO-DAG: %[[SparseToSparse:.*]] = arith.constant 3 : i32
//   CHECK-AUTO-DAG: %[[P:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-AUTO-DAG: %[[Q:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[R:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-AUTO-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<1xi8> to memref<?xi8>
//   CHECK-AUTO-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<1xindex> to memref<?xindex>
//   CHECK-AUTO-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<1xindex> to memref<?xindex>
//       CHECK-AUTO: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %[[SparseToSparse]], %[[A]])
//       CHECK-AUTO: return %[[T]] : !llvm.ptr<i8>
func.func @sparse_convert_singleton(%arg0: tensor<?xf32, #SparseSingleton64>) -> tensor<?xf32, #SparseSingleton32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseSingleton64> to tensor<?xf32, #SparseSingleton32>
  return %0 : tensor<?xf32, #SparseSingleton32>
}
