// RUN: mlir-opt %s -canonicalize | FileCheck %s

// Test case: Basic folding of tensor_load(tensor_to_memref(t)) -> t
// CHECK-LABEL:   func @tensor_load_of_tensor_to_memref(
// CHECK-SAME:                                          %[[TENSOR:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           return %[[TENSOR]]
func @tensor_load_of_tensor_to_memref(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = tensor_to_memref %arg0 : memref<?xf32>
    %1 = tensor_load %0 : memref<?xf32>
    return %1 : tensor<?xf32>
}

// Test case: Basic folding of tensor_to_memref(tensor_load(m)) -> m
// CHECK-LABEL:   func @tensor_to_memref_of_tensor_load(
// CHECK-SAME:                                          %[[MEMREF:.*]]: memref<?xf32>) -> memref<?xf32> {
// CHECK:           return %[[MEMREF]]
func @tensor_to_memref_of_tensor_load(%arg0: memref<?xf32>) -> memref<?xf32> {
    %0 = tensor_load %arg0 : memref<?xf32>
    %1 = tensor_to_memref %0 : memref<?xf32>
    return %1 : memref<?xf32>
}

// Test case: If the memrefs are not the same type, don't fold them.
// CHECK-LABEL:   func @no_fold_tensor_to_memref_of_tensor_load(
// CHECK-SAME:                                                  %[[MEMREF_ADDRSPACE2:.*]]: memref<?xf32, 2>) -> memref<?xf32, 7> {
// CHECK:           %[[TENSOR:.*]] = tensor_load %[[MEMREF_ADDRSPACE2]] : memref<?xf32, 2>
// CHECK:           %[[MEMREF_ADDRSPACE7:.*]] = tensor_to_memref %[[TENSOR]] : memref<?xf32, 7>
// CHECK:           return %[[MEMREF_ADDRSPACE7]]
func @no_fold_tensor_to_memref_of_tensor_load(%arg0: memref<?xf32, 2>) -> memref<?xf32, 7> {
    %0 = tensor_load %arg0 : memref<?xf32, 2>
    %1 = tensor_to_memref %0 : memref<?xf32, 7>
    return %1 : memref<?xf32, 7>
}

// Test case: Basic folding of dim(tensor_load(m)) -> dim(m).
// CHECK-LABEL: func @dim_of_tensor_load(
//  CHECK-SAME:     %[[MEMREF:[0-9a-z]*]]: memref<?xf32>
//       CHECK:   %[[C0:.*]] = constant 0
//       CHECK:   %[[D:.*]] = dim %[[MEMREF]], %[[C0]]
//       CHECK:   return %[[D]] : index
func @dim_of_tensor_load(%arg0: memref<?xf32>) -> index {
    %c0 = constant 0 : index
    %0 = tensor_load %arg0 : memref<?xf32>
    %1 = dim %0, %c0 : tensor<?xf32>
    return %1 : index
}
