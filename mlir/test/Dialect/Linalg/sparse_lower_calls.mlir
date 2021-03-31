// RUN: mlir-opt --test-sparsification="lower" %s | FileCheck %s

!SparseTensor = type !llvm.ptr<i8>

// CHECK-LABEL: func @sparse_pointers(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = constant 1 : index
//       CHECK: %[[T:.*]] = call @sparsePointers64(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func @sparse_pointers(%arg0: !SparseTensor) -> memref<?xindex> {
  %a = linalg.sparse_tensor %arg0 : !SparseTensor to tensor<128xf64>
  %c = constant 1 : index
  %0 = linalg.sparse_pointers %a, %c : tensor<128xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// CHECK-LABEL: func @sparse_pointers32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = constant 1 : index
//       CHECK: %[[T:.*]] = call @sparsePointers32(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xi32>
//       CHECK: return %[[T]] : memref<?xi32>
func @sparse_pointers32(%arg0: !SparseTensor) -> memref<?xi32> {
  %a = linalg.sparse_tensor %arg0 : !SparseTensor to tensor<128xf64>
  %c = constant 1 : index
  %0 = linalg.sparse_pointers %a, %c : tensor<128xf64> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_indices(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = constant 1 : index
//       CHECK: %[[T:.*]] = call @sparseIndices64(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func @sparse_indices(%arg0: !SparseTensor) -> memref<?xindex> {
  %a = linalg.sparse_tensor %arg0 : !SparseTensor to tensor<128xf64>
  %c = constant 1 : index
  %0 = linalg.sparse_indices %a, %c : tensor<128xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// CHECK-LABEL: func @sparse_indices32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = constant 1 : index
//       CHECK: %[[T:.*]] = call @sparseIndices32(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xi32>
//       CHECK: return %[[T]] : memref<?xi32>
func @sparse_indices32(%arg0: !SparseTensor) -> memref<?xi32> {
  %a = linalg.sparse_tensor %arg0 : !SparseTensor to tensor<128xf64>
  %c = constant 1 : index
  %0 = linalg.sparse_indices %a, %c : tensor<128xf64> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_valuesf64(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = call @sparseValuesF64(%[[A]]) : (!llvm.ptr<i8>) -> memref<?xf64>
//       CHECK: return %[[T]] : memref<?xf64>
func @sparse_valuesf64(%arg0: !SparseTensor) -> memref<?xf64> {
  %a = linalg.sparse_tensor %arg0 : !SparseTensor to tensor<128xf64>
  %0 = linalg.sparse_values %a : tensor<128xf64> to memref<?xf64>
  return %0 : memref<?xf64>
}

// CHECK-LABEL: func @sparse_valuesf32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = call @sparseValuesF32(%[[A]]) : (!llvm.ptr<i8>) -> memref<?xf32>
//       CHECK: return %[[T]] : memref<?xf32>
func @sparse_valuesf32(%arg0: !SparseTensor) -> memref<?xf32> {
  %a = linalg.sparse_tensor %arg0 : !SparseTensor to tensor<128xf32>
  %0 = linalg.sparse_values %a : tensor<128xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}
