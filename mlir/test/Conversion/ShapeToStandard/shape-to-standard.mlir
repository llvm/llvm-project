// RUN: mlir-opt --split-input-file --convert-shape-to-std --verify-diagnostics %s | FileCheck %s --dump-input-on-failure

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
