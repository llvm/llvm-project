// RUN: mlir-opt -split-input-file %s | FileCheck %s

!SparseTensor = type !llvm.ptr<i8>

// CHECK-LABEL: func @sparse_tensor(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = linalg.sparse_tensor %[[A]] : !llvm.ptr<i8> to tensor<128xf64>
//       CHECK: return %[[T]] : tensor<128xf64>
func @sparse_tensor(%arg0: !SparseTensor) -> tensor<128xf64> {
  %0 = linalg.sparse_tensor %arg0 : !SparseTensor to tensor<128xf64>
  return %0 : tensor<128xf64>
}

// -----

// CHECK-LABEL: func @sparse_pointers(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64>)
//       CHECK: %[[C:.*]] = constant 1 : index
//       CHECK: %[[T:.*]] = linalg.sparse_pointers %[[A]], %[[C]] : tensor<128xf64> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func @sparse_pointers(%arg0: tensor<128xf64>) -> memref<?xindex> {
  %c = constant 1 : index
  %0 = linalg.sparse_pointers %arg0, %c : tensor<128xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

// CHECK-LABEL: func @sparse_indices(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64>)
//       CHECK: %[[C:.*]] = constant 1 : index
//       CHECK: %[[T:.*]] = linalg.sparse_indices %[[A]], %[[C]] : tensor<128xf64> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func @sparse_indices(%arg0: tensor<128xf64>) -> memref<?xindex> {
  %c = constant 1 : index
  %0 = linalg.sparse_indices %arg0, %c : tensor<128xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

// CHECK-LABEL: func @sparse_values(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64>)
//       CHECK: %[[T:.*]] = linalg.sparse_values %[[A]] : tensor<128xf64> to memref<?xf64>
//       CHECK: return %[[T]] : memref<?xf64>
func @sparse_values(%arg0: tensor<128xf64>) -> memref<?xf64> {
  %0 = linalg.sparse_values %arg0 : tensor<128xf64> to memref<?xf64>
  return %0 : memref<?xf64>
}
