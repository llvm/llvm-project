// RUN: mlir-opt %s -test-vector-dim-info-propagation | FileCheck %s
// RUN: mlir-opt %s -canonicalize -test-vector-dim-info-propagation | FileCheck %s

// Check that canonicalization keeps per-dim information. The pass attaches a
// `test.dim_map` attribute to vector.transpose, vector.broadcast,
// vector.shape_cast and return ops. It says where the information of each dim
// comes from:
//   * d1: argument dim d1.
//   * n: a new dim, added by a broadcast or a shape_cast.
//   * 0: undefined.
//
// Both RUN lines share the checks. The ones with a FIXME fail right now and
// expose cases where an existing canonicalization drops information.

///----------------------------------------------------------------------------
/// vector.transpose
///----------------------------------------------------------------------------

// The transpose moves data, so nothing folds.
// CHECK-LABEL: @transpose_non_unit_dims(
// CHECK: return {test.dim_map = "(d1, d0)"}
func.func @transpose_non_unit_dims(%arg0: vector<2x3xf32>) -> vector<3x2xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// Only a unit dim moves. Same map as @shape_cast_one_unit_dim.
// CHECK-LABEL: @transpose_one_unit_dim(
// CHECK: return {test.dim_map = "(d1, d0)"}
func.func @transpose_one_unit_dim(%arg0: vector<4x1xf32>) -> vector<1x4xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<4x1xf32> to vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

// Two unit dims. A shape_cast would not say which is which.
// CHECK-LABEL: @transpose_several_unit_dims(
// CHECK: return {test.dim_map = "(d1, d0, d2)"}
func.func @transpose_several_unit_dims(%arg0: vector<4x1x1xf32>) -> vector<1x4x1xf32> {
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<4x1x1xf32> to vector<1x4x1xf32>
  return %0 : vector<1x4x1xf32>
}

// FIXME: folded away, so the information of d1 and d2 is no longer swapped.
// CHECK-LABEL: @transpose_unit_dims_same_type(
// CHECK: return {test.dim_map = "(d0, d2, d1)"}
func.func @transpose_unit_dims_same_type(%arg0: vector<4x1x1xf32>) -> vector<4x1x1xf32> {
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<4x1x1xf32> to vector<4x1x1xf32>
  return %0 : vector<4x1x1xf32>
}

// The transposes cancel out.
// CHECK-LABEL: @transpose_of_transpose_cancelled(
// CHECK: return {test.dim_map = "(d0, d1, d2)"}
func.func @transpose_of_transpose_cancelled(%arg0: vector<2x3x4xf32>) -> vector<2x3x4xf32> {
  %0 = vector.transpose %arg0, [1, 2, 0] : vector<2x3x4xf32> to vector<3x4x2xf32>
  %1 = vector.transpose %0, [2, 0, 1] : vector<3x4x2xf32> to vector<2x3x4xf32>
  return %1 : vector<2x3x4xf32>
}

// The transposes fold into one.
// CHECK-LABEL: @transpose_of_transpose_composed(
// CHECK: return {test.dim_map = "(d2, d0, d1)"}
func.func @transpose_of_transpose_composed(%arg0: vector<2x3x4xf32>) -> vector<4x2x3xf32> {
  %0 = vector.transpose %arg0, [1, 2, 0] : vector<2x3x4xf32> to vector<3x4x2xf32>
  %1 = vector.transpose %0, [1, 2, 0] : vector<3x4x2xf32> to vector<4x2x3xf32>
  return %1 : vector<4x2x3xf32>
}

///----------------------------------------------------------------------------
/// vector.shape_cast
///----------------------------------------------------------------------------

// One unit dim on each side, so its information moves with it.
// CHECK-LABEL: @shape_cast_one_unit_dim(
// CHECK: return {test.dim_map = "(d1, d0)"}
func.func @shape_cast_one_unit_dim(%arg0: vector<4x1xf32>) -> vector<1x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<4x1xf32> to vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

// Two unit dims on each side, so their information is undefined.
// CHECK-LABEL: @shape_cast_several_unit_dims(
// CHECK: return {test.dim_map = "({{.*}}, d0, {{.*}})"}
func.func @shape_cast_several_unit_dims(%arg0: vector<4x1x1xf32>) -> vector<1x4x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<4x1x1xf32> to vector<1x4x1xf32>
  return %0 : vector<1x4x1xf32>
}

// No unit dim in the source, so both unit dims are new.
// CHECK-LABEL: @shape_cast_adds_unit_dims(
// CHECK: return {test.dim_map = "(d0, n, n)"}
func.func @shape_cast_adds_unit_dims(%arg0: vector<4xf32>) -> vector<4x1x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<4xf32> to vector<4x1x1xf32>
  return %0 : vector<4x1x1xf32>
}

// Nothing says which unit dim has the information of d1.
// CHECK-LABEL: @shape_cast_adds_unit_dim_next_to_unit_dim(
// CHECK: return {test.dim_map = "(d0, {{.*}}, {{.*}})"}
func.func @shape_cast_adds_unit_dim_next_to_unit_dim(%arg0: vector<4x1xf32>) -> vector<4x1x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<4x1xf32> to vector<4x1x1xf32>
  return %0 : vector<4x1x1xf32>
}

// The unit dim that is left could have the information of d0 or d2.
// CHECK-LABEL: @shape_cast_removes_unit_dim(
// CHECK: return {test.dim_map = "({{.*}}, d1)"}
func.func @shape_cast_removes_unit_dim(%arg0: vector<1x4x1xf32>) -> vector<1x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<1x4x1xf32> to vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

// Folds to the argument, which retrieves the information of the unit dim.
// CHECK-LABEL: @shape_cast_of_shape_cast_folded_away(
// CHECK: return {test.dim_map = "(d0, {{.*}})"}
func.func @shape_cast_of_shape_cast_folded_away(%arg0: vector<4x1xf32>) -> vector<4x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<4x1xf32> to vector<4x1x1xf32>
  %1 = vector.shape_cast %0 : vector<4x1x1xf32> to vector<4x1xf32>
  return %1 : vector<4x1xf32>
}

// FIXME: folds to the argument, where the new unit dim has the information of
// d0.
// CHECK-LABEL: @shape_cast_of_shape_cast_removes_and_adds(
// CHECK: return {test.dim_map = "(n, d1)"}
func.func @shape_cast_of_shape_cast_removes_and_adds(%arg0: vector<1x4xf32>) -> vector<1x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<1x4xf32> to vector<4xf32>
  %1 = vector.shape_cast %0 : vector<4xf32> to vector<1x4xf32>
  return %1 : vector<1x4xf32>
}

///----------------------------------------------------------------------------
/// vector.broadcast
///----------------------------------------------------------------------------

// Two new dims. d0 and d1 keep their information.
// CHECK-LABEL: @broadcast_adds_keeps_expands(
// CHECK: return {test.dim_map = "(n, n, d0, d1)"}
func.func @broadcast_adds_keeps_expands(%arg0: vector<4x1xf32>) -> vector<8x6x4x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<8x6x4x2xf32>
  return %0 : vector<8x6x4x2xf32>
}

// The broadcasts fold into one with both new dims.
// CHECK-LABEL: @broadcast_of_broadcast(
// CHECK: return {test.dim_map = "(n, n, d0)"}
func.func @broadcast_of_broadcast(%arg0: vector<7xf32>) -> vector<2x3x7xf32> {
  %0 = vector.broadcast %arg0 : vector<7xf32> to vector<3x7xf32>
  %1 = vector.broadcast %0 : vector<3x7xf32> to vector<2x3x7xf32>
  return %1 : vector<2x3x7xf32>
}

// Becomes a shape_cast with the same new dims.
// CHECK-LABEL: @broadcast_adds_unit_dims(
// CHECK: return {test.dim_map = "(n, n, d0)"}
func.func @broadcast_adds_unit_dims(%arg0: vector<4xf32>) -> vector<1x1x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<1x1x4xf32>
  return %0 : vector<1x1x4xf32>
}

// FIXME: becomes a shape_cast, which does not say which unit dim has the
// information of d0.
// CHECK-LABEL: @broadcast_adds_unit_dim_next_to_unit_dim(
// CHECK: return {test.dim_map = "(n, d0, d1)"}
func.func @broadcast_adds_unit_dim_next_to_unit_dim(%arg0: vector<1x4xf32>) -> vector<1x1x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<1x1x4xf32>
  return %0 : vector<1x1x4xf32>
}

///----------------------------------------------------------------------------
/// vector.transpose and vector.shape_cast
///----------------------------------------------------------------------------

// One unit dim, so both ops say where its information goes.
// CHECK-LABEL: @shape_cast_transpose_one_unit_dim(
// CHECK: return {test.dim_map = "(d0, d2, d1)"}
func.func @shape_cast_transpose_one_unit_dim(%arg0: vector<2x1x3xf32>) -> vector<2x3x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x1x3xf32> to vector<1x2x3xf32>
  %1 = vector.transpose %0, [1, 2, 0] : vector<1x2x3xf32> to vector<2x3x1xf32>
  return %1 : vector<2x3x1xf32>
}

// CHECK-LABEL: @transpose_shape_cast_one_unit_dim(
// CHECK: return {test.dim_map = "(d0, d2, d1)"}
func.func @transpose_shape_cast_one_unit_dim(%arg0: vector<2x1x3xf32>) -> vector<2x3x1xf32> {
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<2x1x3xf32> to vector<1x2x3xf32>
  %1 = vector.shape_cast %0 : vector<1x2x3xf32> to vector<2x3x1xf32>
  return %1 : vector<2x3x1xf32>
}

// The chain is the identity and folds to the argument.
// CHECK-LABEL: @shape_cast_transpose_one_unit_dim_folded_away(
// CHECK: return {test.dim_map = "(d0, d1)"}
func.func @shape_cast_transpose_one_unit_dim_folded_away(%arg0: vector<4x1xf32>) -> vector<4x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<4x1xf32> to vector<1x4xf32>
  %1 = vector.transpose %0, [1, 0] : vector<1x4xf32> to vector<4x1xf32>
  return %1 : vector<4x1xf32>
}

// Only new unit dims move, so the fold into one shape_cast drops nothing.
// CHECK-LABEL: @shape_cast_transpose_added_unit_dims(
// CHECK: return {test.dim_map = "(n, d0, n)"}
func.func @shape_cast_transpose_added_unit_dims(%arg0: vector<4xf32>) -> vector<1x4x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<4xf32> to vector<4x1x1xf32>
  %1 = vector.transpose %0, [2, 0, 1] : vector<4x1x1xf32> to vector<1x4x1xf32>
  return %1 : vector<1x4x1xf32>
}

// Only a new unit dim moves, so the fold into one shape_cast drops nothing.
// CHECK-LABEL: @shape_cast_transpose_added_unit_dim(
// CHECK: return {test.dim_map = "(n, d0, d1)"}
func.func @shape_cast_transpose_added_unit_dim(%arg0: vector<2x3xf32>) -> vector<1x2x3xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x3xf32> to vector<2x1x3xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<2x1x3xf32> to vector<1x2x3xf32>
  return %1 : vector<1x2x3xf32>
}

// The unit dim information is undefined either way, so the fold is fine.
// CHECK-LABEL: @shape_cast_transpose_several_unit_dims(
// CHECK: return {test.dim_map = "(d1, {{.*}}, {{.*}})"}
func.func @shape_cast_transpose_several_unit_dims(%arg0: vector<1x4xf32>) -> vector<4x1x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<1x4xf32> to vector<1x4x1xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<1x4x1xf32> to vector<4x1x1xf32>
  return %1 : vector<4x1x1xf32>
}

// CHECK-LABEL: @transpose_shape_cast_several_unit_dims(
// CHECK: return {test.dim_map = "({{.*}}, d0)"}
func.func @transpose_shape_cast_several_unit_dims(%arg0: vector<4x1x1xf32>) -> vector<1x4xf32> {
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<4x1x1xf32> to vector<1x4x1xf32>
  %1 = vector.shape_cast %0 : vector<1x4x1xf32> to vector<1x4xf32>
  return %1 : vector<1x4xf32>
}

// Folds to the argument, which retrieves the unit dim information.
// CHECK-LABEL: @shape_cast_transpose_several_unit_dims_folded_away(
// CHECK: return {test.dim_map = "({{.*}}, d1, {{.*}})"}
func.func @shape_cast_transpose_several_unit_dims_folded_away(%arg0: vector<1x4x1xf32>) -> vector<1x4x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<1x4x1xf32> to vector<4x1x1xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<4x1x1xf32> to vector<1x4x1xf32>
  return %1 : vector<1x4x1xf32>
}

// The transpose moves data, so nothing folds.
// CHECK-LABEL: @transpose_non_unit_dims_shape_cast(
// CHECK: return {test.dim_map = "(d1, d0, n)"}
func.func @transpose_non_unit_dims_shape_cast(%arg0: vector<2x3xf32>) -> vector<3x2x1xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  %1 = vector.shape_cast %0 : vector<3x2xf32> to vector<3x2x1xf32>
  return %1 : vector<3x2x1xf32>
}

///----------------------------------------------------------------------------
/// vector.transpose and vector.broadcast
///----------------------------------------------------------------------------

// No broadcast puts a new dim last, so nothing folds.
// CHECK-LABEL: @broadcast_transpose_non_unit_dim(
// CHECK: return {test.dim_map = "(d0, n)"}
func.func @broadcast_transpose_non_unit_dim(%arg0: vector<7xf32>) -> vector<7x2xf32> {
  %0 = vector.broadcast %arg0 : vector<7xf32> to vector<2x7xf32>
  %1 = vector.transpose %0, [1, 0] : vector<2x7xf32> to vector<7x2xf32>
  return %1 : vector<7x2xf32>
}

// Only new dims move, so the fold into one broadcast drops nothing.
// CHECK-LABEL: @broadcast_transpose_added_dims(
// CHECK: return {test.dim_map = "(n, n, d0)"}
func.func @broadcast_transpose_added_dims(%arg0: vector<7xf32>) -> vector<3x2x7xf32> {
  %0 = vector.broadcast %arg0 : vector<7xf32> to vector<2x3x7xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<2x3x7xf32> to vector<3x2x7xf32>
  return %1 : vector<3x2x7xf32>
}

// FIXME: the folded broadcast swaps the information of d0 and d1.
// CHECK-LABEL: @broadcast_transpose_expanded_dims(
// CHECK: return {test.dim_map = "(d1, d0, d2)"}
func.func @broadcast_transpose_expanded_dims(%arg0: vector<1x1x7xf32>) -> vector<3x2x7xf32> {
  %0 = vector.broadcast %arg0 : vector<1x1x7xf32> to vector<2x3x7xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<2x3x7xf32> to vector<3x2x7xf32>
  return %1 : vector<3x2x7xf32>
}

// FIXME: in the folded broadcast d0 and the new dim trade places.
// CHECK-LABEL: @broadcast_transpose_unit_dim_and_added_dim(
// CHECK: return {test.dim_map = "(d0, n)"}
func.func @broadcast_transpose_unit_dim_and_added_dim(%arg0: vector<1xf32>) -> vector<1x8xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<8x1xf32>
  %1 = vector.transpose %0, [1, 0] : vector<8x1xf32> to vector<1x8xf32>
  return %1 : vector<1x8xf32>
}

// FIXME: in the folded broadcast d0 and the new dim trade places.
// CHECK-LABEL: @broadcast_transpose_added_dim_and_expanded_dim(
// CHECK: return {test.dim_map = "(d0, n, d1)"}
func.func @broadcast_transpose_added_dim_and_expanded_dim(%arg0: vector<1x7xf32>) -> vector<2x3x7xf32> {
  %0 = vector.broadcast %arg0 : vector<1x7xf32> to vector<3x2x7xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<3x2x7xf32> to vector<2x3x7xf32>
  return %1 : vector<2x3x7xf32>
}

// FIXME: the folded broadcast swaps the information of d0 and d1.
// CHECK-LABEL: @broadcast_transpose_unit_dim_and_expanded_dim(
// CHECK: return {test.dim_map = "(d1, d0)"}
func.func @broadcast_transpose_unit_dim_and_expanded_dim(%arg0: vector<1x1xf32>) -> vector<1x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x1xf32> to vector<4x1xf32>
  %1 = vector.transpose %0, [1, 0] : vector<4x1xf32> to vector<1x4xf32>
  return %1 : vector<1x4xf32>
}

// Only a new dim moves, so the fold into one shape_cast drops nothing.
// CHECK-LABEL: @broadcast_transpose_added_unit_dims(
// CHECK: return {test.dim_map = "(n, d0, n)"}
func.func @broadcast_transpose_added_unit_dims(%arg0: vector<4xf32>) -> vector<1x4x1xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<1x1x4xf32>
  %1 = vector.transpose %0, [0, 2, 1] : vector<1x1x4xf32> to vector<1x4x1xf32>
  return %1 : vector<1x4x1xf32>
}

// FIXME: becomes a shape_cast, which does not say where the information of d0
// goes.
// CHECK-LABEL: @broadcast_transpose_added_unit_dim_next_to_unit_dim(
// CHECK: return {test.dim_map = "(n, d1, d0)"}
func.func @broadcast_transpose_added_unit_dim_next_to_unit_dim(%arg0: vector<1x4xf32>) -> vector<1x4x1xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<1x1x4xf32>
  %1 = vector.transpose %0, [0, 2, 1] : vector<1x1x4xf32> to vector<1x4x1xf32>
  return %1 : vector<1x4x1xf32>
}

///----------------------------------------------------------------------------
/// vector.broadcast and vector.shape_cast
///----------------------------------------------------------------------------

// Folds into one broadcast with the same map.
// CHECK-LABEL: @broadcast_shape_cast_added_outer_unit_dim(
// CHECK: return {test.dim_map = "(n, n, d0)"}
func.func @broadcast_shape_cast_added_outer_unit_dim(%arg0: vector<4xf32>) -> vector<1x2x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<2x4xf32>
  %1 = vector.shape_cast %0 : vector<2x4xf32> to vector<1x2x4xf32>
  return %1 : vector<1x2x4xf32>
}

// Folds into one broadcast with the same map.
// CHECK-LABEL: @broadcast_shape_cast_added_inner_unit_dim(
// CHECK: return {test.dim_map = "(n, n, d0)"}
func.func @broadcast_shape_cast_added_inner_unit_dim(%arg0: vector<4xf32>) -> vector<2x1x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<2x4xf32>
  %1 = vector.shape_cast %0 : vector<2x4xf32> to vector<2x1x4xf32>
  return %1 : vector<2x1x4xf32>
}

// Folds into one broadcast with the same map.
// CHECK-LABEL: @broadcast_shape_cast_removed_unit_dim(
// CHECK: return {test.dim_map = "(n, d0)"}
func.func @broadcast_shape_cast_removed_unit_dim(%arg0: vector<4xf32>) -> vector<2x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<2x1x4xf32>
  %1 = vector.shape_cast %0 : vector<2x1x4xf32> to vector<2x4xf32>
  return %1 : vector<2x4xf32>
}

// No broadcast of the argument expands d1 in front of d0, so nothing folds.
// CHECK-LABEL: @shape_cast_broadcast_one_unit_dim(
// CHECK: return {test.dim_map = "(d1, d0)"}
func.func @shape_cast_broadcast_one_unit_dim(%arg0: vector<4x1xf32>) -> vector<3x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<4x1xf32> to vector<1x4xf32>
  %1 = vector.broadcast %0 : vector<1x4xf32> to vector<3x4xf32>
  return %1 : vector<3x4xf32>
}

// The broadcast expands the new dim. Folds into one broadcast.
// CHECK-LABEL: @shape_cast_broadcast_expanded_added_dim(
// CHECK: return {test.dim_map = "(n, d0)"}
func.func @shape_cast_broadcast_expanded_added_dim(%arg0: vector<4xf32>) -> vector<3x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<4xf32> to vector<1x4xf32>
  %1 = vector.broadcast %0 : vector<1x4xf32> to vector<3x4xf32>
  return %1 : vector<3x4xf32>
}

// FIXME: folds into one broadcast, where the new unit dim has the information
// of d0.
// CHECK-LABEL: @shape_cast_broadcast_removed_unit_dim(
// CHECK: return {test.dim_map = "(n, n, d1)"}
func.func @shape_cast_broadcast_removed_unit_dim(%arg0: vector<1x4xf32>) -> vector<3x1x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<1x4xf32> to vector<4xf32>
  %1 = vector.broadcast %0 : vector<4xf32> to vector<3x1x4xf32>
  return %1 : vector<3x1x4xf32>
}

///----------------------------------------------------------------------------
/// vector.shape_cast, vector.broadcast and vector.transpose
///----------------------------------------------------------------------------

// Not folded today. One broadcast would give the new dim the information of d2.
// CHECK-LABEL: @shape_cast_broadcast_transpose(
// CHECK: return {test.dim_map = "({{.*}}, d1, n)"}
func.func @shape_cast_broadcast_transpose(%arg0: vector<1x32x1xf32>) -> vector<1x32x64xf32> {
  %0 = vector.shape_cast %arg0 : vector<1x32x1xf32> to vector<1x32xf32>
  %1 = vector.broadcast %0 : vector<1x32xf32> to vector<64x1x32xf32>
  %2 = vector.transpose %1, [1, 2, 0] : vector<64x1x32xf32> to vector<1x32x64xf32>
  return %2 : vector<1x32x64xf32>
}

// Only new dims move, so the fold into one broadcast drops nothing.
// CHECK-LABEL: @broadcast_shape_cast_transpose_added_dims(
// CHECK: return {test.dim_map = "(n, n, d0)"}
func.func @broadcast_shape_cast_transpose_added_dims(%arg0: vector<4xf32>) -> vector<1x2x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<2x4xf32>
  %1 = vector.shape_cast %0 : vector<2x4xf32> to vector<2x1x4xf32>
  %2 = vector.transpose %1, [1, 0, 2] : vector<2x1x4xf32> to vector<1x2x4xf32>
  return %2 : vector<1x2x4xf32>
}
