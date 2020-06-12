// RUN: mlir-opt -split-input-file -allow-unregistered-dialect -canonicalize <%s | FileCheck %s

// CHECK-LABEL: func @f
func @f(%arg0: tensor<2x3x4xf32>) -> !shape.shape {
  // CHECK: shape.const_shape [2, 3, 4]
  %0 = "shape.shape_of"(%arg0) : (tensor<2x3x4xf32>) -> !shape.shape
  return %0 : !shape.shape
}

// -----

// Basic case.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.const_shape [2, 3]
  // CHECK: shape.const_shape [4, 5]
  %c2 = constant 2 : i32
  %0 = shape.const_shape [2, 3, 4, 5]
  %head, %tail = "shape.split_at"(%0, %c2) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape

}

// -----

// Negative split point.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.const_shape [2, 3, 4]
  // CHECK: shape.const_shape [5]
  %c-1 = constant -1 : i32
  %0 = shape.const_shape [2, 3, 4, 5]
  %head, %tail = "shape.split_at"(%0, %c-1) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape
}

// -----

// Out of range split point. No folding.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.split_at
  %c5 = constant 5 : i32
  %0 = shape.const_shape [2, 3, 4, 5]
  %head, %tail = "shape.split_at"(%0, %c5) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape
}

// -----

// Basic case.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.const_shape [7, 2]
  %0 = shape.const_shape [1, 2]
  %1 = shape.const_shape [7, 1]
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  return %2 : !shape.shape
}

// -----

// Incompatible shapes. No folding.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [2]
  %1 = shape.const_shape [7]
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  return %2 : !shape.shape
}

// -----

// Basic case.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.const_shape [0, 1, 2, 3]
  %lhs = shape.const_shape [0, 1]
  %rhs = shape.const_shape [2, 3]
  %0 = "shape.concat"(%lhs, %rhs) : (!shape.shape, !shape.shape) -> !shape.shape
  return %0 : !shape.shape
}

// -----

// Basic case.
// CHECK-LABEL: func @f
func @f() -> tensor<2xindex> {
  // CHECK: constant dense<[0, 1]> : tensor<2xindex>
  %cs = shape.const_shape [0, 1]
  %0 = "shape.to_extent_tensor"(%cs) : (!shape.shape) -> tensor<2xindex>
  return %0 : tensor<2xindex>
}

// -----

// Basic case.
// CHECK-LABEL: func @f()
func @f() -> !shape.shape {
  // CHECK: shape.const_shape [3, 5, 11]
  %e0 = constant 3 : index
  %e1 = constant 5 : index
  %e2 = constant 11 : index
  %ret = shape.from_extents %e0, %e1, %e2
  return %ret : !shape.shape
}

// -----

// CHECK-LABEL: func @no_fold
func @no_fold(%arg0: index) -> !shape.shape {
  // CHECK-NOT: shape.const_shape
  %e0 = constant 3 : index
  %ret = shape.from_extents %e0, %arg0
  return %ret : !shape.shape
}

// -----

// Cast constant size to index and fold it away.
// CHECK-LABEL: func @const_size_to_index
func @const_size_to_index() -> index {
  // CHECK-NOT: shape.index_cast
  %cs = shape.const_size 123
  // CHECK: constant 123 : index
  %ci = shape.size_to_index %cs
  return %ci : index
}

// -----

// Cast constant index to size and fold it away.
// CHECK-LABEL: func @const_index_to_size
func @const_index_to_size() -> !shape.size {
  // CHECK-NOT: index_cast
  %ci = constant 123 : index
  // CHECK: shape.const_size 123
  %cs = shape.index_to_size %ci
  return %cs : !shape.size
}

// -----

// Cast constant index to size, then back, and fold it away.
// CHECK-LABEL: func @const_index_to_size_to_index
func @const_index_to_size_to_index() -> index {
  // CHECK-NOT: shape.index_cast
  %ci0 = constant 123 : index
  %cs0 = shape.index_to_size %ci0
  // CHECK: %[[CI:.*]] = constant 123 : index
  // CHECK-NEXT: return %[[CI]] : index
  %ci1 = shape.size_to_index %cs0
  return %ci1 : index
}

// -----

// No folding.
// CHECK-LABEL: func @nonfoldable_size_to_index
func @nonfoldable_size_to_index(%cs : !shape.size) -> index {
  // CHECK: shape.size_to_index
  %ci = shape.size_to_index %cs
  return %ci : index
}

// -----

// No folding.
// CHECK-LABEL: func @nonfoldable_index_to_size
func @nonfoldable_index_to_size(%ci : index) -> !shape.size {
  // CHECK: shape.index_to_size
  %cs = shape.index_to_size %ci
  return %cs : !shape.size
}

// -----

// Fold number of elements computation.
// CHECK-LABEL: func @num_elements
func @num_elements() -> !shape.size {
  // CHECK-NOT: shape.const_shape
  %shape = shape.const_shape [4, 5, 6]
  // CHECK-NOT: shape.num_elements
  %num_elements = shape.num_elements %shape
  // CHECK: %[[NUM:.*]] = shape.const_size 120
  // CHECK-NEXT: return %[[NUM]] : !shape.size
  return %num_elements : !shape.size
}

// -----

// No folding.
// CHECK-LABEL: func @nonfoldable_num_elements
func @nonfoldable_num_elements(%shape : !shape.shape) -> !shape.size {
  // CHECK-NOT: shape.const_{{.*}}
  %num_elements = shape.num_elements %shape
  return %num_elements : !shape.size
}

// -----

// Basic folding.
// CHECK-LABEL: func @basic
func @basic() -> !shape.size {
  // CHECK: shape.const_size 2
  %0 = shape.const_shape [0, 1, 2]
  %c2 = shape.const_size 2
  %1 = shape.get_extent %0, %c2
  return %1 : !shape.size
}

// -----

// Should not fold.
// CHECK-LABEL: func @out_of_bounds
func @out_of_bounds() -> !shape.size {
  // CHECK: shape.const_shape
  // CHECK: shape.get_extent
  %0 = shape.const_shape [0, 1, 2]
  %c3 = shape.const_size 3
  %1 = shape.get_extent %0, %c3
  return %1 : !shape.size
}

// -----

// Should not fold.
// CHECK-LABEL: func @not_const
func @not_const(%arg0: !shape.shape) -> !shape.size {
  // CHECK: shape.get_extent
  %c3 = shape.const_size 3
  %0 = shape.get_extent %arg0, %c3
  return %0 : !shape.size
}


// -----
// cstr_eq with non-constant but known equal shapes can be removed.
// CHECK-LABEL: func @f
func @f(%arg0 : !shape.shape) {
  // CHECK-NEXT: shape.const_witness true
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.cstr_eq %arg0, %arg0, %arg0
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----
// cstr_eq with equal const_shapes can be folded
// CHECK-LABEL: func @f
func @f() {
  // CHECK-NEXT: shape.const_witness true
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %cs0 = shape.const_shape [0, 1]
  %cs1 = shape.const_shape [0, 1]
  %cs2 = shape.const_shape [0, 1]
  %0 = shape.cstr_eq %cs0, %cs1, %cs2
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----
// cstr_eq with unequal const_shapes cannot be folded
// CHECK-LABEL: func @f
func @f() {
  // CHECK-NEXT: shape.const_shape
  // CHECK-NEXT: shape.const_shape
  // CHECK-NEXT: shape.cstr_eq
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %cs0 = shape.const_shape [0, 1]
  %cs1 = shape.const_shape [3, 1]
  %0 = shape.cstr_eq %cs0, %cs1
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----
// cstr_eq without const_shapes cannot be folded
// CHECK-LABEL: func @f
func @f(%arg0: !shape.shape, %arg1: !shape.shape) {
  // CHECK-NEXT: shape.cstr_eq
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.cstr_eq %arg0, %arg1
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----

// assuming_all with known passing witnesses can be folded
// CHECK-LABEL: func @f
func @f() {
  // CHECK-NEXT: shape.const_witness true
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.const_witness true
  %1 = shape.const_witness true
  %2 = shape.const_witness true
  %3 = shape.assuming_all %0, %1, %2
  "consume.witness"(%3) : (!shape.witness) -> ()
  return
}

// -----

// assuming_all should not be removed if not all witnesses are statically passing.
//
// Additionally check that the attribute is moved to the end as this op is
// commutative.
// CHECK-LABEL: func @f
func @f() {
  // CHECK-NEXT: %[[UNKNOWN:.*]] = "test.source"
  // CHECK-NEXT: shape.assuming_all %[[UNKNOWN]]
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.const_witness true
  %1 = "test.source"() : () -> !shape.witness
  %2 = shape.assuming_all %0, %1
  "consume.witness"(%2) : (!shape.witness) -> ()
  return
}

// -----

// any can be replaced with a constant input if it has one.
// CHECK-LABEL: func @f
func @f(%arg0 : !shape.shape) -> !shape.shape {
  // CHECK-NEXT: %[[CS:.*]] = shape.const_shape
  // CHECK-NEXT: return %[[CS]]
  %0 = shape.const_shape [2, 3, 4]
  %1 = shape.any %0, %arg0
  return %1 : !shape.shape
}


// -----

// Folding of any with partially constant operands is not yet implemented.
// CHECK-LABEL: func @f
func @f(%arg0 : !shape.shape, %arg1 : !shape.shape) -> !shape.shape {
  // CHECK-NEXT: %[[CS:.*]] = shape.any
  // CHECK-NEXT: return %[[CS]]
  %1 = shape.any %arg0, %arg1
  return %1 : !shape.shape
}

// -----

// assuming with a known passing witness can be removed
// CHECK-LABEL: func @f
func @f() {
  // CHECK-NEXT: source
  // CHECK-NEXT: sink
  // CHECK-NEXT: return
  %0 = shape.const_witness true
  %1 = shape.assuming %0 -> index {
    %2 = "test.source"() : () -> (index)
    shape.assuming_yield %2 : index
  }
  "test.sink"(%1) : (index) -> ()
  return
}

// -----

// assuming without a known passing passing witness cannot be removed
// CHECK-LABEL: func @f
func @f() {
  // CHECK-NEXT: test.source
  // CHECK-NEXT: shape.assuming
  // CHECK-NEXT:   test.source
  // CHECK-NEXT:   shape.assuming_yield
  // CHECK-NEXT: }
  // CHECK-NEXT: test.sink
  // CHECK-NEXT: return
  %0 = "test.source"() : () -> (!shape.witness)
  %1 = shape.assuming %0 -> index {
    %2 = "test.source"() : () -> (index)
    shape.assuming_yield %2 : index
  }
  "test.sink"(%1) : (index) -> ()
  return
}

// -----
// Broadcastable with broadcastable constant shapes can be removed.
// CHECK-LABEL: func @f
func @f() {
  // CHECK-NEXT: shape.const_witness true
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %cs0 = shape.const_shape [3, 1]
  %cs1 = shape.const_shape [1, 5]
  %0 = shape.cstr_broadcastable %cs0, %cs1
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----
// Broadcastable with non-broadcastable constant shapes is always false
// CHECK-LABEL: func @f
func @f() {
  // CHECK-NEXT: shape.const_shape
  // CHECK-NEXT: shape.const_shape
  // CHECK-NEXT: shape.cstr_broadcastable
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %cs0 = shape.const_shape [1, 3]
  %cs1 = shape.const_shape [1, 5]
  %0 = shape.cstr_broadcastable %cs0, %cs1
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----
// Broadcastable without guaranteed broadcastable shapes cannot be removed.
// CHECK-LABEL: func @f
func @f(%arg0 : !shape.shape) {
  // CHECK-NEXT: shape.const_shape
  // CHECK-NEXT: shape.cstr_broadcastable
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %cs0 = shape.const_shape [1,3]
  %0 = shape.cstr_broadcastable %arg0, %cs0
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----
// Broadcastable with non-constant but known equal shapes can be removed.
// CHECK-LABEL: func @f
func @f(%arg0 : !shape.shape) {
  // CHECK-NEXT: shape.const_witness true
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.cstr_broadcastable %arg0, %arg0
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}
