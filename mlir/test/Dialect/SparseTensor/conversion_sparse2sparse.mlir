// First use with `kViaCOO` for sparse2sparse conversion (the old way).
// RUN: mlir-opt %s --sparse-tensor-conversion="s2s-strategy=1" \
// RUN:    --canonicalize --cse | FileCheck %s -check-prefix=CHECK-COO
//
// Now again with `kAuto` (the new default).
// RUN: mlir-opt %s --sparse-tensor-conversion="s2s-strategy=0" \
// RUN:    --canonicalize --cse | FileCheck %s -check-prefix=CHECK-AUTO

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
func.func @sparse_convert(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}
