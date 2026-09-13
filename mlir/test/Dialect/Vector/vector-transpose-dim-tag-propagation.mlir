// RUN: mlir-opt %s -canonicalize -test-vector-transpose-dim-tag-propagation | FileCheck %s

// Test that canonicalization of `vector.transpose`, `vector.broadcast` and
// `vector.shape_cast` does not drop per-dim information. The
// test-vector-transpose-dim-tag-propagation pass propagates the `dim_tags`
// attribute, standing in for per-dim information such as a layout or
// distribution axis, forward through these ops. The CHECK lines verify that
// propagation still succeeds after canonicalization.

// CHECK-LABEL: @propagate_through_transpose
// CHECK: vector.transpose
// CHECK-SAME: propagated_tags = array<i64: 222, 111>
func.func @propagate_through_transpose(%arg0: vector<2x3xf32>) -> vector<3x2xf32> {
  %0 = vector.transpose %arg0, [1, 0] {dim_tags = array<i64: 111, 222>}
      : vector<2x3xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// CHECK-LABEL: @propagate_through_shape_cast
// CHECK: vector.shape_cast
// CHECK-SAME: propagated_tags = array<i64: 111, 111, 222, 222>
func.func @propagate_through_shape_cast(%arg0: vector<4x6xf32>) -> vector<1x4x6x1xf32> {
  %0 = vector.shape_cast %arg0 {dim_tags = array<i64: 111, 222>}
      : vector<4x6xf32> to vector<1x4x6x1xf32>
  return %0 : vector<1x4x6x1xf32>
}

// Broadcast is trailing aligned: prepended dims are new and have no source
// tag, stretched unit dims keep theirs.
// CHECK-LABEL: @propagate_through_broadcast
// CHECK: vector.broadcast
// CHECK-SAME: propagated_tags = array<i64: -1, 111, 222>
func.func @propagate_through_broadcast(%arg0: vector<4x1xf32>) -> vector<8x4x2xf32> {
  %0 = vector.broadcast %arg0 {dim_tags = array<i64: 111, 222>}
      : vector<4x1xf32> to vector<8x4x2xf32>
  return %0 : vector<8x4x2xf32>
}

// Merging dims picks the leftmost source tag.
// CHECK-LABEL: @propagate_through_shape_cast_merge
// CHECK: vector.shape_cast
// CHECK-SAME: propagated_tags = array<i64: 111>
func.func @propagate_through_shape_cast_merge(%arg0: vector<2x4xf32>) -> vector<8xf32> {
  %0 = vector.shape_cast %arg0 {dim_tags = array<i64: 111, 222>}
      : vector<2x4xf32> to vector<8xf32>
  return %0 : vector<8xf32>
}

// Splitting a dim propagates its tag to all the resulting dims.
// CHECK-LABEL: @propagate_through_shape_cast_split
// CHECK: vector.shape_cast
// CHECK-SAME: propagated_tags = array<i64: 111, 111>
func.func @propagate_through_shape_cast_split(%arg0: vector<8xf32>) -> vector<2x4xf32> {
  %0 = vector.shape_cast %arg0 {dim_tags = array<i64: 111>}
      : vector<8xf32> to vector<2x4xf32>
  return %0 : vector<2x4xf32>
}

// Folding this transpose into the shape_cast would drop its permutation and
// the unit dim tags with it. Transposing with [1, 0, 2] instead would fold to
// the same shape_cast, so the tags are unrecoverable once folded.
// CHECK-LABEL: @unit_dim_transpose_folded_into_shape_cast
// CHECK: vector.transpose
// CHECK-SAME: propagated_tags = array<i64: 333, 111, 222>
func.func @unit_dim_transpose_folded_into_shape_cast(%arg0: vector<4xf32>) -> vector<1x4x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<4xf32> to vector<4x1x1xf32>
  %1 = vector.transpose %0, [2, 0, 1] {dim_tags = array<i64: 111, 222, 333>}
      : vector<4x1x1xf32> to vector<1x4x1xf32>
  return %1 : vector<1x4x1xf32>
}

// Folding this transpose into the broadcast would drop which broadcast dim
// each tag belongs to.
// CHECK-LABEL: @broadcast_dims_transposed
// CHECK: vector.transpose
// CHECK-SAME: propagated_tags = array<i64: 222, 111, 333>
func.func @broadcast_dims_transposed(%arg0: vector<7xf32>) -> vector<3x2x7xf32> {
  %0 = vector.broadcast %arg0 : vector<7xf32> to vector<2x3x7xf32>
  %1 = vector.transpose %0, [1, 0, 2] {dim_tags = array<i64: 111, 222, 333>}
      : vector<2x3x7xf32> to vector<3x2x7xf32>
  return %1 : vector<3x2x7xf32>
}
