// RUN: mlir-opt --split-input-file --convert-shape-to-std --verify-diagnostics %s | FileCheck %s

// Convert `size` to `index` type.
// CHECK-LABEL: @size_id
// CHECK-SAME: (%[[SIZE:.*]]: index)
func @size_id(%size : !shape.size) -> !shape.size {
  // CHECK: return %[[SIZE]] : index
  return %size : !shape.size
}

// -----

// Convert `shape` to `tensor<?xindex>` type.
// CHECK-LABEL: @shape_id
// CHECK-SAME: (%[[SHAPE:.*]]: tensor<?xindex>)
func @shape_id(%shape : !shape.shape) -> !shape.shape {
  // CHECK: return %[[SHAPE]] : tensor<?xindex>
  return %shape : !shape.shape
}

// -----

// Lower binary ops.
// CHECK-LABEL: @binary_ops
// CHECK-SAME: (%[[LHS:.*]]: index, %[[RHS:.*]]: index)
func @binary_ops(%lhs : !shape.size, %rhs : !shape.size) {
  // CHECK: addi %[[LHS]], %[[RHS]] : index
  %sum = "shape.add"(%lhs, %rhs) : (!shape.size, !shape.size) -> !shape.size
  return
}

// -----

// Lower binary ops.
// CHECK-LABEL: @binary_ops
// CHECK-SAME: (%[[LHS:.*]]: index, %[[RHS:.*]]: index)
func @binary_ops(%lhs : index, %rhs : index) {
  // CHECK: muli %[[LHS]], %[[RHS]] : index
  %product = shape.mul %lhs, %rhs : index, index -> index
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

