// RUN: mlir-opt --split-input-file --convert-shape-to-std --verify-diagnostics %s | FileCheck %s

// Lower binary ops.
// CHECK-LABEL: @binary_ops
// CHECK-SAME: (%[[LHS:.*]]: index, %[[RHS:.*]]: index)
func @binary_ops(%lhs : index, %rhs : index) {
  // CHECK: addi %[[LHS]], %[[RHS]] : index
  %sum = shape.add %lhs, %rhs : index, index -> index
  // CHECK: muli %[[LHS]], %[[RHS]] : index
  %product = shape.mul %lhs, %rhs : index, index -> index
  return
}

// -----

// Don't lower binary ops when they operate on `shape.size`.
// CHECK-LABEL: @binary_ops_on_size
// CHECK-SAME: (%[[LHS:.*]]: !shape.size, %[[RHS:.*]]: !shape.size)
func @binary_ops_on_size(%lhs : !shape.size, %rhs : !shape.size) {
  // CHECK: shape.add %[[LHS]], %[[RHS]] : !shape.size, !shape.size -> !shape.size
  // CHECK: shape.mul %[[LHS]], %[[RHS]] : !shape.size, !shape.size -> !shape.size
  %sum = shape.add %lhs, %rhs : !shape.size, !shape.size -> !shape.size
  %prod = shape.mul %lhs, %rhs : !shape.size, !shape.size -> !shape.size
  return
}

// -----

// Don't lower `shape_of` with `shape.shape` type.
// CHECK-LABEL: @shape_of
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x2x3xf32>)
func @shape_of_stat(%arg : tensor<1x2x3xf32>) {
  // CHECK: shape.shape_of %[[ARG]] : tensor<1x2x3xf32> -> !shape.shape
  %shape = shape.shape_of %arg : tensor<1x2x3xf32> -> !shape.shape
  return
}

// -----

// Lower `shape_of` for statically shaped tensor.
// CHECK-LABEL: @shape_of_stat
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x2x3xf32>)
func @shape_of_stat(%arg : tensor<1x2x3xf32>) {
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = constant 3 : index
  // CHECK-DAG: %[[SHAPE_UNCASTED:.*]] = tensor_from_elements(%[[C1]], %[[C2]], %[[C3]]) : tensor<3xindex>
  %shape = shape.shape_of %arg : tensor<1x2x3xf32> -> tensor<?xindex>
  return
}

// -----

// Lower `shape_of` for dynamically shaped tensor.
// CHECK-LABEL: @shape_of_dyn
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x5x?xf32>)
func @shape_of_dyn(%arg : tensor<1x5x?xf32>) {
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index
  // CHECK-DAG: %[[C5:.*]] = constant 5 : index
  // CHECK-DAG: %[[C2:.*]] = constant 2 : index
  // CHECK-DAG: %[[DYN_DIM:.*]] = dim %[[ARG]], %[[C2]] : tensor<1x5x?xf32>
  // CHECK-DAG: %[[SHAPE_UNCASTED:.*]] = tensor_from_elements(%[[C1]], %[[C5]], %[[DYN_DIM]]) : tensor<3xindex>
  %shape = shape.shape_of %arg : tensor<1x5x?xf32> -> tensor<?xindex>
  return
}

// -----

// Convert `rank` to `dim` of the first dimension.
// CHECK-LABEL: @rank
// CHECK-SAME: (%[[SHAPE:.*]]: tensor<?xindex>) -> index
func @rank(%shape : tensor<?xindex>) -> index {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[RESULT:.*]] = dim %[[SHAPE]], %[[C0]]
  // CHECK: return %[[RESULT]] : index
  %rank = shape.rank %shape : tensor<?xindex> -> index
  return %rank : index
}

// -----

// Don't lower `get_extent` if it is of type `shape.size`.
// CHECK-LABEL: @get_extent
func @get_extent(%shape : tensor<?xindex>, %idx : !shape.size) -> !shape.size {
  // CHECK: shape.get_extent
  %result = shape.get_extent %shape, %idx
      : tensor<?xindex>, !shape.size -> !shape.size
  return %result : !shape.size
}

// -----

// Don't lower `rank` if type is not error-free.
// CHECK-LABEL: @rank
func @rank(%shape : !shape.shape) {
  // CHECK: shape.rank
  %rank = shape.rank %shape : !shape.shape -> !shape.size
  return
}

// -----

// Express `get_extent` as `std.dim` when it relies directly on the outcome of a
// `shape_of` operation.
// CHECK-LABEL: @get_extent_shape_of
// CHECK-SAME:  (%[[ARG:.*]]: tensor<2x3xf32>, %[[IDX:.*]]: index) -> index
func @get_extent_shape_of(%arg : tensor<2x3xf32>, %idx : index) -> index {
  // CHECK: %[[RESULT:.*]] = dim %[[ARG]], %[[IDX]] : tensor<2x3xf32>
  // CHECK: return %[[RESULT]] : index
  %shape = shape.shape_of %arg : tensor<2x3xf32> -> tensor<?xindex>
  %result = shape.get_extent %shape, %idx : tensor<?xindex>, index -> index
  return %result : index
}

// -----

// Express `get_extent` as `std.extract_element`.
// CHECK-LABEL: @get_extent_from_extent_tensor
// CHECK-SAME: (%[[EXTENTS:.*]]: tensor<?xindex>, %[[IDX:.*]]: index) -> index
func @get_extent_from_extent_tensor(%extents : tensor<?xindex>, %idx : index)
    -> index {
  // CHECK: %[[RESULT:.*]] = extract_element %[[EXTENTS]][%[[IDX]]] : tensor<?xindex>
  // CHECK: return %[[RESULT]] : index
  %result = shape.get_extent %extents, %idx : tensor<?xindex>, index -> index
  return %result : index
}

// -----

// Lower `const_shape` to `tensor_from_elements`.
// CHECK-LABEL: @const_shape
// CHECK-SAME: () -> tensor<?xindex>
func @const_shape() -> tensor<?xindex> {
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: %[[C3:.*]] = constant 3 : index
  // CHECK: %[[TENSOR3:.*]] = tensor_from_elements(%[[C1]], %[[C2]], %[[C3]])
  // CHECK: %[[RESULT:.*]] = tensor_cast %[[TENSOR3]] : tensor<3xindex> to tensor<?xindex>
  // CHECK: return %[[RESULT]] : tensor<?xindex>
  %shape = shape.const_shape [1, 2, 3] : tensor<?xindex>
  return %shape : tensor<?xindex>
}

// -----

// Lower `any` to its first operand.
// CHECK-LABEL: @any_of_three
// CHECK-SAME:  (%[[A:.*]]: tensor<?xindex>, %[[B:.*]]: tensor<?xindex>, %[[C:.*]]: tensor<?xindex>) -> tensor<?xindex>
func @any_of_three(%a : tensor<?xindex>,
                   %b : tensor<?xindex>,
                   %c : tensor<?xindex>) -> tensor<?xindex> {
  // CHECK: return %[[A]] : tensor<?xindex>
  %result = "shape.any"(%a, %b, %c) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>) -> tensor<?xindex>
  return %result : tensor<?xindex>
}

// -----

// Lower `any` to its first operand.
// CHECK-LABEL: @any_of_one
// CHECK-SAME:  (%[[A:.*]]: tensor<?xindex>) -> tensor<?xindex>
func @any_of_one(%a : tensor<?xindex>) -> tensor<?xindex> {
  // CHECK: return %[[A]] : tensor<?xindex>
  %result = "shape.any"(%a) : (tensor<?xindex>) -> tensor<?xindex>
  return %result : tensor<?xindex>
}

// -----

// Lower 'const_size` to `std.constant`
// CHECK-LABEL: @const_size
func @const_size() -> index {
  // CHECK: %[[RES:.*]] = constant 42 : index
  %size = shape.const_size 42
  %result = shape.size_to_index %size : !shape.size
  // CHECK: return %[[RES]]
  return %result : index
}

// -----

// Lower `to_extent_tensor` to `std.tensor_cast`
// Fold to_extent_tensor when already on tensor.
// CHECK-LABEL: @to_extent_tensor
// CHECK-SAME: (%[[ARG:.*]]: tensor<?xindex>
func @to_extent_tensor(%arg: tensor<?xindex>) -> tensor<3xindex> {
  // CHECK-NOT: to_extent_tensor
  // CHECK: %[[RES:.*]] = tensor_cast %[[ARG]] : tensor<?xindex> to tensor<3xindex
  %casted = shape.to_extent_tensor %arg : tensor<?xindex> -> tensor<3xindex>
  // CHECK: return %[[RES]]
  return %casted : tensor<3xindex>
}
