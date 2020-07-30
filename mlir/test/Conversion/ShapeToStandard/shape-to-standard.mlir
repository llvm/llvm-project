// RUN: mlir-opt --split-input-file --convert-shape-to-std --verify-diagnostics %s | FileCheck %s

// Convert `size` to `index` type.
// CHECK-LABEL: @size_id
// CHECK-SAME: (%[[SIZE:.*]]: index)
func @size_id(%size : !shape.size) -> !shape.size {
  // CHECK: return %[[SIZE]] : index
  return %size : !shape.size
}

// -----

// Lower `size_to_index` conversion to no-op.
// CHECK-LABEL: @size_to_index
// CHECK-SAME: (%[[SIZE:.*]]: index) -> index
func @size_to_index(%size : !shape.size) -> index {
  // CHECK-NEXT: return %[[SIZE]] : index
  %index = shape.size_to_index %size
  return %index : index
}

// -----

// Lower `index_to_size` conversion to no-op.
// CHECK-LABEL: @index_to_size
// CHECK-SAME: (%[[INDEX:.*]]: index) -> index
func @index_to_size(%index : index) -> !shape.size {
  // CHECK-NEXT: return %[[INDEX]] : index
  %size = shape.index_to_size %index
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

// Lower `to_extent_tensor` operation to no-op.
// CHECK-LABEL: @to_extent_tensor
// CHECK-SAME: (%[[SHAPE:.*]]: tensor<?xindex>) -> tensor<?xindex>
func @to_extent_tensor(%shape : !shape.shape) -> tensor<?xindex> {
  // CHECK-NEXT: return %[[SHAPE]] : tensor<?xindex>
  %tensor = "shape.to_extent_tensor"(%shape) : (!shape.shape) -> tensor<?xindex>
  return %tensor : tensor<?xindex>
}

// -----

// Lower `from_extent_tensor` operation to no-op.
// CHECK-LABEL: @from_extent_tensor
// CHECK-SAME: (%[[TENSOR:.*]]: tensor<?xindex>) -> tensor<?xindex>
func @from_extent_tensor(%tensor : tensor<?xindex>) -> !shape.shape {
  // CHECK-NEXT: return %[[TENSOR]] : tensor<?xindex>
  %shape = "shape.from_extent_tensor"(%tensor)
      : (tensor<?xindex>) -> !shape.shape
  return %shape : !shape.shape
}

// -----

// Lower binary ops.
// CHECK-LABEL: @binary_ops
// CHECK-SAME: (%[[LHS:.*]]: index, %[[RHS:.*]]: index)
func @binary_ops(%lhs : !shape.size, %rhs : !shape.size) {
  %sum = "shape.add"(%lhs, %rhs) : (!shape.size, !shape.size) -> !shape.size
  // CHECK-NEXT: addi %[[LHS]], %[[RHS]] : index
  %product = shape.mul %lhs, %rhs
  // CHECK-NEXT: muli %[[LHS]], %[[RHS]] : index
  return
}

// -----

// Convert `const_size` to `constant` op.
// CHECK-LABEL: @size_const
func @size_const() -> !shape.size {
  %c1 = shape.const_size 1
  return %c1 : !shape.size
}
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: return %[[C1]] : index
// -----

// Lower `shape_of` for statically shaped tensor.
// CHECK-LABEL: @shape_of_stat
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x2x3xf32>)
func @shape_of_stat(%arg : tensor<1x2x3xf32>) {
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = constant 3 : index
  // CHECK-DAG: %[[SHAPE:.*]] = tensor_from_elements(%[[C1]], %[[C2]], %[[C3]]) : tensor<3xindex>
  %shape = shape.shape_of %arg : tensor<1x2x3xf32>
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
  // CHECK-DAG: %[[SHAPE:.*]] = tensor_from_elements(%[[C1]], %[[C5]], %[[DYN_DIM]]) : tensor<3xindex>
  %shape = shape.shape_of %arg : tensor<1x5x?xf32>
  return
}

// -----

// Convert `rank` to `dim` of the first dimension.
// CHECK-LABEL: @rank
// CHECK-SAME: (%[[SHAPE:.*]]: tensor<?xindex>) -> index
func @rank(%shape : !shape.shape) -> !shape.size {
  // CHECK-DAG: %[[C0:.*]] = constant 0 : index
  // CHECK-DAG: %[[RESULT:.*]] = dim %[[SHAPE]], %[[C0]]
  // CHECK-DAG: return %[[RESULT]] : index
  %rank = shape.rank %shape
  return %rank : !shape.size
}

// -----

// Express `get_extent` as `std.dim` when it relies directly on the outcome of a
// `shape_of` operation.
// CHECK-LABEL: @get_extent_shape_of
// CHECK-SAME:  (%[[ARG:.*]]: tensor<2x3xf32>, %[[IDX:.*]]: index) -> index
func @get_extent_shape_of(%arg : tensor<2x3xf32>, %idx : !shape.size)
    -> !shape.size {
  // CHECK: %[[RESULT:.*]] = dim %[[ARG]], %[[IDX]] : tensor<2x3xf32>
  // CHECK: return %[[RESULT]] : index
  %shape = shape.shape_of %arg : tensor<2x3xf32>
  %result = shape.get_extent %shape, %idx
  return %result : !shape.size
}

// -----

// Express `get_extent` as `std.extract_element` when it relies directly on the
// outcome of a `from_extent_tensor` operation.
// CHECK-LABEL: @get_extent_from_extent_tensor
// CHECK-SAME: (%[[EXTENTS:.*]]: tensor<?xindex>, %[[IDX:.*]]: index) -> index
func @get_extent_from_extent_tensor(%extents : tensor<?xindex>,
                                    %idx : !shape.size) -> !shape.size {
  // CHECK: %[[RESULT:.*]] = extract_element %[[EXTENTS]][%[[IDX]]] : tensor<?xindex>
  // CHECK: return %[[RESULT]] : index
  %shape = shape.from_extent_tensor %extents : tensor<?xindex>
  %result = shape.get_extent %shape, %idx
  return %result : !shape.size
}

