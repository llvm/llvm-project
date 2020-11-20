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

// Test case: Folding of dim(dynamic_tensor_from_elements %idx) -> %idx
// CHECK-LABEL: func @dim_of_dynamic_tensor_from_elements(
//  CHECK-SAME:     %[[IDX0:[0-9a-z]+]]: index, %[[IDX1:[0-9a-z]+]]: index
//   CHECK-NOT:   dim
//       CHECK:   return %[[IDX1]] : index
func @dim_of_dynamic_tensor_from_elements(%arg0: index, %arg1: index) -> index {
  %c3 = constant 3 : index
  %0 = dynamic_tensor_from_elements %arg0, %arg1 {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
    yield %c3 : index
  } : tensor<2x?x4x?x5xindex>
  %1 = dim %0, %c3 : tensor<2x?x4x?x5xindex>
  return %1 : index
}

// Test case: Folding of comparisons with equal operands.
// CHECK-LABEL: @cmpi_equal_operands
//   CHECK-DAG:   %[[T:.*]] = constant true
//   CHECK-DAG:   %[[F:.*]] = constant false
//       CHECK:   return %[[T]], %[[T]], %[[T]], %[[T]], %[[T]],
//  CHECK-SAME:          %[[F]], %[[F]], %[[F]], %[[F]], %[[F]]
func @cmpi_equal_operands(%arg0: i64)
    -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %0 = cmpi "eq", %arg0, %arg0 : i64
  %1 = cmpi "sle", %arg0, %arg0 : i64
  %2 = cmpi "sge", %arg0, %arg0 : i64
  %3 = cmpi "ule", %arg0, %arg0 : i64
  %4 = cmpi "uge", %arg0, %arg0 : i64
  %5 = cmpi "ne", %arg0, %arg0 : i64
  %6 = cmpi "slt", %arg0, %arg0 : i64
  %7 = cmpi "sgt", %arg0, %arg0 : i64
  %8 = cmpi "ult", %arg0, %arg0 : i64
  %9 = cmpi "ugt", %arg0, %arg0 : i64
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
      : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}
