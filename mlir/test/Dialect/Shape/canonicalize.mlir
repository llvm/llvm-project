// RUN: mlir-opt -split-input-file -allow-unregistered-dialect -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @f
func @f(%arg0: tensor<2x3x4xf32>) -> tensor<?xindex> {
  // CHECK: shape.const_shape [2, 3, 4] : tensor<?xindex>
  %0 = shape.shape_of %arg0 : tensor<2x3x4xf32> -> tensor<?xindex>
  return %0 : tensor<?xindex>
}

// -----

// Basic case.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.const_shape [2, 3] : !shape.shape
  // CHECK: shape.const_shape [4, 5] : !shape.shape
  %c2 = constant 2 : i32
  %0 = shape.const_shape [2, 3, 4, 5] : !shape.shape
  %head, %tail = "shape.split_at"(%0, %c2) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape

}

// -----

// Negative split point.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.const_shape [2, 3, 4] : !shape.shape
  // CHECK: shape.const_shape [5] : !shape.shape
  %c-1 = constant -1 : i32
  %0 = shape.const_shape [2, 3, 4, 5] : !shape.shape
  %head, %tail = "shape.split_at"(%0, %c-1) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape
}

// -----

// Out of range split point. No folding.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.split_at
  %c5 = constant 5 : i32
  %0 = shape.const_shape [2, 3, 4, 5] : !shape.shape
  %head, %tail = "shape.split_at"(%0, %c5) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape
}

// -----

// Basic case.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.const_shape [7, 2] : !shape.shape
  %0 = shape.const_shape [1, 2] : !shape.shape
  %1 = shape.const_shape [7, 1] : !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}

// -----

// Basic case including extent tensors.
// CHECK-LABEL: @broadcast
func @broadcast() -> tensor<?xindex> {
  // CHECK: shape.const_shape [7, 2] : tensor<?xindex>
  %0 = shape.const_shape [1, 2] : tensor<?xindex>
  %1 = shape.const_shape [7, 1] : tensor<?xindex>
  %2 = shape.broadcast %0, %1
      : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return %2 : tensor<?xindex>
}

// -----

// Basic case including extent tensors.
// CHECK-LABEL: @broadcast
func @broadcast() -> !shape.shape {
  // CHECK: shape.const_shape [7, 2] : !shape.shape
  %0 = shape.const_shape [1, 2] : tensor<?xindex>
  %1 = shape.const_shape [7, 1] : tensor<?xindex>
  %2 = shape.broadcast %0, %1 : tensor<?xindex>, tensor<?xindex> -> !shape.shape
  return %2 : !shape.shape
}

// -----

// Rhs is a scalar.
// CHECK-LABEL: func @f
func @f(%arg0 : !shape.shape) -> !shape.shape {
  // CHECK: return %arg0
  %0 = shape.const_shape [] : !shape.shape
  %1 = shape.broadcast %arg0, %0 : !shape.shape, !shape.shape -> !shape.shape
  return %1 : !shape.shape
}

// -----

// Lhs is a scalar.
// CHECK-LABEL: func @f
func @f(%arg0 : !shape.shape) -> !shape.shape {
  // CHECK: return %arg0
  %0 = shape.const_shape [] : !shape.shape
  %1 = shape.broadcast %0, %arg0 : !shape.shape, !shape.shape -> !shape.shape
  return %1 : !shape.shape
}

// -----

// Lhs is a scalar and rhs is constant.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: %[[CST:.*]] = shape.const_shape [1, 2, 3] : !shape.shape
  // CHECK: return %[[CST]]
  %0 = shape.const_shape [] : !shape.shape
  %1 = shape.const_shape [1, 2, 3] : !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}

// -----

// Incompatible shapes. No folding.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [2] : !shape.shape
  %1 = shape.const_shape [7] : !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}

// -----

// Basic case.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.const_shape [0, 1, 2, 3] : !shape.shape
  %lhs = shape.const_shape [0, 1] : !shape.shape
  %rhs = shape.const_shape [2, 3] : !shape.shape
  %0 = shape.concat %lhs, %rhs
  return %0 : !shape.shape
}

// -----

// Basic case.
// CHECK-LABEL: func @f
func @f() -> tensor<2xindex> {
  // CHECK: constant dense<[0, 1]> : tensor<2xindex>
  %cs = shape.const_shape [0, 1] : !shape.shape
  %0 = shape.to_extent_tensor %cs : !shape.shape -> tensor<2xindex>
  return %0 : tensor<2xindex>
}

// -----

// Basic case.
// CHECK-LABEL: func @f()
func @f() -> !shape.shape {
  // CHECK: shape.const_shape [3, 5, 11] : !shape.shape
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
  %ci = shape.size_to_index %cs : !shape.size
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
  %ci1 = shape.size_to_index %cs0 : !shape.size
  return %ci1 : index
}

// -----

// No folding.
// CHECK-LABEL: func @nonfoldable_size_to_index
func @nonfoldable_size_to_index(%cs : !shape.size) -> index {
  // CHECK: shape.size_to_index
  %ci = shape.size_to_index %cs : !shape.size
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
  %shape = shape.const_shape [4, 5, 6] : !shape.shape
  // CHECK-NOT: shape.num_elements
  %num_elements = shape.num_elements %shape : !shape.shape -> !shape.size
  // CHECK: %[[NUM:.*]] = shape.const_size 120
  // CHECK-NEXT: return %[[NUM]] : !shape.size
  return %num_elements : !shape.size
}

// -----

// No folding.
// CHECK-LABEL: func @nonfoldable_num_elements
func @nonfoldable_num_elements(%shape : !shape.shape) -> !shape.size {
  // CHECK-NOT: shape.const_{{.*}}
  %num_elements = shape.num_elements %shape : !shape.shape -> !shape.size
  return %num_elements : !shape.size
}

// -----

// Basic folding.
// CHECK-LABEL: func @basic
func @basic() -> index {
  // CHECK: constant 2 : index
  %0 = shape.const_shape [0, 1, 2] : tensor<?xindex>
  %c2 = constant 2 : index
  %1 = shape.get_extent %0, %c2 : tensor<?xindex>, index -> index
  return %1 : index
}

// -----

// Should not fold.
// CHECK-LABEL: func @out_of_bounds
func @out_of_bounds() -> index {
  // CHECK: shape.const_shape
  // CHECK: shape.get_extent
  %0 = shape.const_shape [0, 1, 2] : tensor<?xindex>
  %c3 = constant 3 : index
  %1 = shape.get_extent %0, %c3 : tensor<?xindex>, index -> index
  return %1 : index
}

// -----

// Should not fold.
// CHECK-LABEL: func @not_const
func @not_const(%arg0: tensor<?xindex>) -> index {
  // CHECK: shape.get_extent
  %c3 = constant 3 : index
  %0 = shape.get_extent %arg0, %c3 : tensor<?xindex>, index -> index
  return %0 : index
}

// -----

// Basic folding.
// CHECK-LABEL: func @basic
func @basic() -> !shape.size {
  // CHECK: shape.const_size 2
  %0 = shape.const_shape [0, 1, 2] : !shape.shape
  %c2 = shape.const_size 2
  %1 = shape.get_extent %0, %c2 : !shape.shape, !shape.size -> !shape.size
  return %1 : !shape.size
}

// -----

// Should not fold.
// CHECK-LABEL: func @out_of_bounds
func @out_of_bounds() -> !shape.size {
  // CHECK: shape.const_shape
  // CHECK: shape.get_extent
  %0 = shape.const_shape [0, 1, 2] : !shape.shape
  %c3 = shape.const_size 3
  %1 = shape.get_extent %0, %c3 : !shape.shape, !shape.size -> !shape.size
  return %1 : !shape.size
}

// -----

// Should not fold.
// CHECK-LABEL: func @not_const
func @not_const(%arg0 : !shape.shape) -> !shape.size {
  // CHECK: shape.get_extent
  %c3 = shape.const_size 3
  %0 = shape.get_extent %arg0, %c3 : !shape.shape, !shape.size -> !shape.size
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
  %cs0 = shape.const_shape [0, 1] : !shape.shape
  %cs1 = shape.const_shape [0, 1] : !shape.shape
  %cs2 = shape.const_shape [0, 1] : !shape.shape
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
  %cs0 = shape.const_shape [0, 1] : !shape.shape
  %cs1 = shape.const_shape [3, 1] : !shape.shape
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
func @f(%arg : !shape.shape) -> !shape.shape {
  // CHECK-NEXT: %[[CS:.*]] = shape.const_shape
  // CHECK-NEXT: return %[[CS]]
  %0 = shape.const_shape [2, 3, 4] : !shape.shape
  %1 = shape.any %0, %arg : !shape.shape, !shape.shape -> !shape.shape
  return %1 : !shape.shape
}

// -----

// any can be replaced with a constant input if it has one.
// CHECK-LABEL: func @f
func @f(%arg : tensor<?xindex>) -> tensor<?xindex> {
  // CHECK-NEXT: %[[CS:.*]] = shape.const_shape [2, 3, 4] : tensor<?xindex>
  // CHECK-NEXT: return %[[CS]] : tensor<?xindex>
  %0 = shape.const_shape [2, 3, 4] : tensor<?xindex>
  %1 = shape.any %0, %arg : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return %1 : tensor<?xindex>
}

// -----

// Folding of any with partially constant operands is not yet implemented.
// CHECK-LABEL: func @f
func @f(%arg0 : !shape.shape, %arg1 : !shape.shape) -> !shape.shape {
  // CHECK-NEXT: %[[CS:.*]] = shape.any
  // CHECK-NEXT: return %[[CS]]
  %1 = shape.any %arg0, %arg1 : !shape.shape, !shape.shape -> !shape.shape
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
  %cs0 = shape.const_shape [3, 1] : !shape.shape
  %cs1 = shape.const_shape [1, 5] : !shape.shape
  %0 = shape.cstr_broadcastable %cs0, %cs1 : !shape.shape, !shape.shape
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----
// Broadcastable with non-broadcastable constant shapes is always false
// CHECK-LABEL: func @static_non_broadcastable
func @static_non_broadcastable() {
  // CHECK-NEXT: shape.const_shape
  // CHECK-NEXT: shape.const_shape
  // CHECK-NEXT: shape.cstr_broadcastable
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %cs0 = shape.const_shape [1, 3] : !shape.shape
  %cs1 = shape.const_shape [1, 5] : !shape.shape
  %0 = shape.cstr_broadcastable %cs0, %cs1 : !shape.shape, !shape.shape
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
  %cs0 = shape.const_shape [1, 3] : !shape.shape
  %0 = shape.cstr_broadcastable %arg0, %cs0 : !shape.shape, !shape.shape
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
  %0 = shape.cstr_broadcastable %arg0, %arg0 : !shape.shape, !shape.shape
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----

// Broadcastable canonicalization also works on extent tensors.
// CHECK-LABEL: func @broadcastable_on_extent_tensors
func @broadcastable_on_extent_tensors(%arg : tensor<?xindex>) {
  // CHECK-NEXT: shape.const_witness true
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.cstr_broadcastable %arg, %arg : tensor<?xindex>, tensor<?xindex>
  "consume.witness"(%0) : (!shape.witness) -> ()
  return
}

// -----

// Fold `rank` based on constant shape.
// CHECK-LABEL: @fold_rank
func @fold_rank() -> !shape.size {
  // CHECK: %[[RESULT:.*]] = shape.const_size 5
  // CHECK: return %[[RESULT]] : !shape.size
  %shape = shape.const_shape [3, 4, 5, 6, 7] : !shape.shape
  %rank = shape.rank %shape : !shape.shape -> !shape.size
  return %rank : !shape.size
}

// -----

// Do not fold `rank` if shape is dynamic.
// CHECK-LABEL: @dont_fold_rank
// CHECK-SAME: (%[[SHAPE:.*]]: !shape.shape) -> !shape.size
func @dont_fold_rank(%shape : !shape.shape) -> !shape.size {
  // CHECK: %[[RESULT:.*]] = shape.rank %[[SHAPE]]
  // CHECK: return %[[RESULT]] : !shape.size
  %rank = shape.rank %shape : !shape.shape -> !shape.size
  return %rank : !shape.size
}

// -----

// Fold `rank` based on constant extent tensor.
// CHECK-LABEL: @fold_rank
func @fold_rank() -> index {
  // CHECK: %[[RESULT:.*]] = constant 5 : index
  // CHECK: return %[[RESULT]] : index
  %shape = shape.const_shape [3, 4, 5, 6, 7] : tensor<?xindex>
  %rank = shape.rank %shape : tensor<?xindex> -> index
  return %rank : index
}

// -----

// Do not fold `rank` for non-constant extent tensors.
// CHECK-LABEL: @dont_fold_rank
// CHECK-SAME: (%[[SHAPE:.*]]: tensor<?xindex>) -> index
func @dont_fold_rank(%shape : tensor<?xindex>) -> index {
  // CHECK: %[[RESULT:.*]] = shape.rank %[[SHAPE]] : tensor<?xindex> -> index
  // CHECK: return %[[RESULT]] : index
  %rank = shape.rank %shape : tensor<?xindex> -> index
  return %rank : index
}

// -----

// Canonicalize `rank` when shape is derived from ranked tensor.
// CHECK-LABEL: @canonicalize_rank
func @canonicalize_rank(%arg : tensor<1x2x?xf32>) -> index {
  // CHECK: %[[RESULT:.*]] = constant 3 : index
  // CHECK: return %[[RESULT]] : index
  %shape = shape.shape_of %arg : tensor<1x2x?xf32> -> tensor<?xindex>
  %rank = shape.rank %shape : tensor<?xindex> -> index
  return %rank : index
}

// -----

// Canonicalize `rank` when shape is derived from ranked tensor.
// CHECK-LABEL: @canonicalize_rank
func @canonicalize_rank_size(%arg : tensor<1x2x?xf32>) -> !shape.size {
  // CHECK: %[[RESULT:.*]] = shape.const_size 3
  // CHECK: return %[[RESULT]] : !shape.size
  %shape = shape.shape_of %arg : tensor<1x2x?xf32> -> !shape.shape
  %rank = shape.rank %shape : !shape.shape -> !shape.size
  return %rank : !shape.size
}

// -----

// Do not canonicalize `rank` when shape is derived from unranked tensor.
// CHECK-LABEL: @dont_canonicalize_rank
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>) -> index
func @dont_canonicalize_rank(%arg : tensor<*xf32>) -> index {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG]] : tensor<*xf32> -> tensor<?xindex>
  // CHECK: %[[SIZE:.*]] = shape.rank %[[SHAPE]]
  // CHECK: return %[[SIZE]] : index
  %shape = shape.shape_of %arg : tensor<*xf32> -> tensor<?xindex>
  %rank = shape.rank %shape : tensor<?xindex> -> index
  return %rank : index
}

// -----

// Canonicalize redundant conversion from `index` to `size` and back.
// CHECK-LABEL: @index_to_size_to_index
// CHECK-SAME: (%[[IDX:.*]]: index) -> index
func @index_to_size_to_index(%index : index) -> index {
  // CHECK: return %[[IDX]] : index
  %size = shape.index_to_size %index
  %result = shape.size_to_index %size : !shape.size
  return %result : index
}

// -----

// Canonicalize redundant conversion from `size` to `index` and back.
// CHECK-LABEL: @size_to_index_to_size
// CHECK-SAME: (%[[SIZE:.*]]: !shape.size) -> !shape.size
func @size_to_index_to_size(%size : !shape.size) -> !shape.size {
  // CHECK: return %[[SIZE]] : !shape.size
  %idx = shape.size_to_index %size : !shape.size
  %result = shape.index_to_size %idx
  return %result : !shape.size
}

// -----

// Canonicalize scalar cstr_broadcastable checks
// CHECK-LABEL: @cstr_broadcastable_scalar
func @cstr_broadcastable_scalar(%arg0 : tensor<?xf32>) {
  // CHECK-NEXT: shape.const_witness true
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.const_shape [] : !shape.shape
  %1 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
  %2 = shape.cstr_broadcastable %0, %1 : !shape.shape, tensor<?xindex>
  "consume.witness"(%2) : (!shape.witness) -> ()
  return
}

// -----

// Do not canonicalize cstr_broadcastable checks with 2 unknowns
// CHECK-LABEL: @cstr_broadcastable_unknown
func @cstr_broadcastable_unknown(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) {
  // CHECK-NEXT: shape.shape_of %arg0
  // CHECK-NEXT: shape.shape_of %arg1
  // CHECK-NEXT: shape.cstr_broadcastable
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<?xindex>
  %2 = shape.cstr_broadcastable %0, %1 : tensor<?xindex>, tensor<?xindex>
  "consume.witness"(%2) : (!shape.witness) -> ()
  return
}

// -----

// Scalars are safe to broadcast to unranked sizes.
// CHECK-LABEL: @cstr_broadcastable_scalar_unranked
func @cstr_broadcastable_scalar_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<index>) {
  // CHECK-NEXT: shape.const_witness true
  // CHECK-NEXT: consume.witness
  // CHECK-NEXT: return
  %0 = shape.shape_of %arg1 : tensor<index> -> tensor<?xindex>
  %1 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %2 = shape.cstr_broadcastable %0, %1 : tensor<?xindex>, tensor<?xindex>
  "consume.witness"(%2) : (!shape.witness) -> ()
  return
}

// -----

// Fold `shape_eq` for equal and constant shapes.
// CHECK-LABEL: @shape_eq_fold_1
func @shape_eq_fold_1() -> i1 {
  // CHECK: %[[RESULT:.*]] = constant true
  // CHECK: return %[[RESULT]] : i1
  %a = shape.const_shape [1, 2, 3] : !shape.shape
  %b = shape.const_shape [1, 2, 3] : tensor<?xindex>
  %result = shape.shape_eq %a, %b : !shape.shape, tensor<?xindex>
  return %result : i1
}

// -----

// Fold `shape_eq` for different but constant shapes of same length.
// CHECK-LABEL: @shape_eq_fold_0
func @shape_eq_fold_0() -> i1 {
  // CHECK: %[[RESULT:.*]] = constant false
  // CHECK: return %[[RESULT]] : i1
  %a = shape.const_shape [1, 2, 3] : tensor<?xindex>
  %b = shape.const_shape [4, 5, 6] : tensor<?xindex>
  %result = shape.shape_eq %a, %b : tensor<?xindex>, tensor<?xindex>
  return %result : i1
}

// -----

// Fold `shape_eq` for different but constant shapes of different length.
// CHECK-LABEL: @shape_eq_fold_0
func @shape_eq_fold_0() -> i1 {
  // CHECK: %[[RESULT:.*]] = constant false
  // CHECK: return %[[RESULT]] : i1
  %a = shape.const_shape [1, 2, 3, 4, 5, 6] : !shape.shape
  %b = shape.const_shape [1, 2, 3] : !shape.shape
  %result = shape.shape_eq %a, %b : !shape.shape, !shape.shape
  return %result : i1
}

// -----

// Do not fold `shape_eq` for non-constant shapes.
// CHECK-LABEL: @shape_eq_do_not_fold
// CHECK-SAME: (%[[A:.*]]: !shape.shape) -> i1
func @shape_eq_do_not_fold(%a : !shape.shape) -> i1 {
  // CHECK: %[[B:.*]] = shape.const_shape [4, 5, 6]
  // CHECK: %[[RESULT:.*]] = shape.shape_eq %[[A]], %[[B]] : !shape.shape, !shape.shape
  // CHECK: return %[[RESULT]] : i1
  %b = shape.const_shape [4, 5, 6] : !shape.shape
  %result = shape.shape_eq %a, %b : !shape.shape, !shape.shape
  return %result : i1
}

// -----

// Fold `mul` for constant sizes.
// CHECK-LABEL: @fold_mul_size
func @fold_mul_size() -> !shape.size {
  // CHECK: %[[RESULT:.*]] = shape.const_size 6
  // CHECK: return %[[RESULT]] : !shape.size
  %c2 = shape.const_size 2
  %c3 = shape.const_size 3
  %result = shape.mul %c2, %c3 : !shape.size, !shape.size -> !shape.size
  return %result : !shape.size
}

// -----

// Fold `mul` for constant indices.
// CHECK-LABEL: @fold_mul_index
func @fold_mul_index() -> index {
  // CHECK: %[[RESULT:.*]] = constant 6 : index
  // CHECK: return %[[RESULT]] : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %result = shape.mul %c2, %c3 : index, index -> index
  return %result : index
}

// -----

// Fold `mul` for mixed constants.
// CHECK-LABEL: @fold_mul_mixed
func @fold_mul_mixed() -> !shape.size {
  // CHECK: %[[RESULT:.*]] = shape.const_size 6
  // CHECK: return %[[RESULT]] : !shape.size
  %c2 = shape.const_size 2
  %c3 = constant 3 : index
  %result = shape.mul %c2, %c3 : !shape.size, index -> !shape.size
  return %result : !shape.size
}

// -----

// Fold index_cast when already on index.
// CHECK-LABEL: @fold_index_cast_on_index
func @fold_index_cast_on_index(%arg: index) -> index {
  // CHECK-NOT: size_to_index
  %casted = shape.size_to_index %arg : index
  return %casted : index
}

// -----

// Fold to_extent_tensor when already on tensor.
// CHECK-LABEL: @fold_to_extent_tensor_on_tensor
func @fold_to_extent_tensor_on_tensor(%arg: tensor<?xindex>) -> tensor<?xindex> {
  // CHECK-NOT: to_extent_tensor
  %casted = shape.to_extent_tensor %arg : tensor<?xindex> -> tensor<?xindex>
  return %casted : tensor<?xindex>
}
