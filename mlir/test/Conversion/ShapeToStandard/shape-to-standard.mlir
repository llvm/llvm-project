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
