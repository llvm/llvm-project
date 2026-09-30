// RUN: mlir-opt %s --split-input-file --canonicalize | FileCheck %s

// This file contains tests where a vector.shape_cast is the result
// of canonicalization.

// **--------------------------------------------------------** //
//   Tests of BroadcastToShapeCast
// **--------------------------------------------------------** //

// CHECK-LABEL: @broadcast_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<4xi8>
//  CHECK-NEXT: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT: return %[[SHAPE_CAST]] : vector<1x1x4xi8>
func.func @broadcast_to_shape_cast(%arg0 : vector<4xi8>) -> vector<1x1x4xi8> {
  %0 = vector.broadcast %arg0 : vector<4xi8> to vector<1x1x4xi8>
  return %0 : vector<1x1x4xi8>
}

// -----

// broadcast can only be transformed to a shape_cast if the number of elements is
// unchanged by the broadcast.
// CHECK-LABEL: @negative_broadcast_increased_elements_to_shape_cast
//   CHECK-NOT: shape_cast
//       CHECK: return
func.func @negative_broadcast_increased_elements_to_shape_cast(%arg0 : vector<1x4xi8>) -> vector<2x3x4xi8> {
  %0 = vector.broadcast %arg0 : vector<1x4xi8> to vector<2x3x4xi8>
  return %0 : vector<2x3x4xi8>
}

// -----

// shape_cast does not support scalar inputs/outputs, so a broadcast of a scalar
// cannot be transformed to a shape_cast.
// CHECK-LABEL: @negative_broadcast_scalar_to_shape_cast
//   CHECK-NOT: shape_cast
//       CHECK: return
func.func @negative_broadcast_scalar_to_shape_cast(%arg0 : i8) -> vector<1xi8> {
  %0 = vector.broadcast %arg0 : i8 to vector<1xi8>
  return %0 : vector<1xi8>
}

// -----

// In this test, broadcast (2)->(1,2,1) is not legal, but shape_cast (2)->(1,2,1) is.
// CHECK-LABEL: func @canonicalize_broadcast_shapecast_to_shapecast
//   CHECK-NOT:   vector.broadcast
//       CHECK:   vector.shape_cast {{.+}} : vector<2xf32> to vector<1x2x1xf32>
func.func @canonicalize_broadcast_shapecast_to_shapecast(%arg0 : vector<2xf32>) -> vector<1x2x1xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<1x2xf32>
  %1 = vector.shape_cast %0 : vector<1x2xf32> to vector<1x2x1xf32>
  return %1 : vector<1x2x1xf32>
}

// -----

// In this test, broadcast (1)->(1,1) and shape_cast (1)->(1,1) are both legal. shape_cast is chosen.
// CHECK-LABEL: func @canonicalize_broadcast_shapecast_both_possible
//   CHECK-NOT:   vector.broadcast
//       CHECK:   vector.shape_cast {{.+}} : vector<1xf32> to vector<1x1xf32>
func.func @canonicalize_broadcast_shapecast_both_possible(%arg0: vector<1xf32>) -> vector<1x1xf32> {
    %0 = vector.broadcast %arg0 : vector<1xf32> to vector<1x1x1xf32>
    %1 = vector.shape_cast %0 : vector<1x1x1xf32> to vector<1x1xf32>
    return %1 : vector<1x1xf32>
}

// -----

// **--------------------------------------------------------** //
//   Tests of ExtractToShapeCast
// **--------------------------------------------------------** //

// CHECK-LABEL: @extract_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<1x4xf32>
//  CHECK-NEXT: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT: return %[[SHAPE_CAST]] : vector<4xf32>
func.func @extract_to_shape_cast(%arg0 : vector<1x4xf32>) -> vector<4xf32> {
  %0 = vector.extract %arg0[0] : vector<4xf32> from vector<1x4xf32>
  return %0 : vector<4xf32>
}

// -----

// In this example, arg1 might be negative indicating poison. We could
// convert this to shape_cast (would be a legal transform with poison)
// but we conservatively choose not to.
// CHECK-LABEL: @negative_extract_to_shape_cast
//   CHECK-NOT: shape_cast
func.func @negative_extract_to_shape_cast(%arg0 : vector<1x4xf32>, %arg1 : index) -> vector<4xf32> {
  %0 = vector.extract %arg0[%arg1] : vector<4xf32> from vector<1x4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extract_shapecast_to_shapecast
//  CHECK-SAME: (%[[ARG:.+]]: vector<3x4xf32>)
//       CHECK:   %[[R:.+]] = vector.shape_cast %[[ARG]] : vector<3x4xf32> to vector<12xf32>
//       CHECK:   return %[[R]]
func.func @fold_extract_shapecast_to_shapecast(%arg0 : vector<3x4xf32>) -> vector<12xf32> {
  %0 = vector.shape_cast %arg0 : vector<3x4xf32> to vector<1x12xf32>
  %r = vector.extract %0[0] : vector<12xf32> from vector<1x12xf32>
  return %r : vector<12xf32>
}

// -----

// CHECK-LABEL: func @insert_extract_to_shape_cast
//  CHECK-SAME: (%[[ARG0:.*]]: vector<1x1x4xf32>, %[[ARG1:.*]]: vector<4xf32>)
//       CHECK:   %[[V0:.*]] = vector.shape_cast %[[ARG0]] : vector<1x1x4xf32> to vector<4xf32>
//       CHECK:   %[[V1:.*]] = vector.shape_cast %[[ARG1]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return %[[V0]], %[[V1]] : vector<4xf32>, vector<1x1x4xf32>
func.func @insert_extract_to_shape_cast(%arg0 : vector<1x1x4xf32>,
  %arg1 : vector<4xf32>) -> (vector<4xf32>, vector<1x1x4xf32>) {
  %0 = vector.extract %arg0[0, 0] : vector<4xf32> from vector<1x1x4xf32>
  %1 = vector.insert %arg1, %arg0 [0, 0] : vector<4xf32> into vector<1x1x4xf32>
  return %0, %1 : vector<4xf32>, vector<1x1x4xf32>
}

// -----

// CHECK-LABEL: func.func @extract_from_broadcast
func.func @extract_from_broadcast(%src: vector<1x1x1xf32>) -> vector<1xf32> {
  %0 = vector.broadcast %src : vector<1x1x1xf32> to vector<1x1x32x1xf32>
  //  CHECK-NEXT:   %[[RES:.*]] = vector.shape_cast{{.*}} vector<1x1x1xf32> to vector<1xf32>
  //  CHECK-NEXT:   return %[[RES]] : vector<1xf32>
  %1 = vector.extract %0[0, 0, 31] : vector<1xf32> from vector<1x1x32x1xf32>
  return %1: vector<1xf32>
}

// -----

// CHECK-LABEL: @no_fold_bcast_mode_switch
// CHECK:         vector.broadcast %{{.*}} : vector<2x1xf32> to vector<2x2x1xf32>
// CHECK-NEXT:    vector.shape_cast %{{.*}} : vector<2x2x1xf32> to vector<2x2xf32>
func.func @no_fold_bcast_mode_switch(%arg0: vector<2x1xf32>) -> vector<2x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2x1xf32> to vector<2x2x1xf32>
  %1 = vector.shape_cast %0 : vector<2x2x1xf32> to vector<2x2xf32>
  return %1 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @no_fold_bcast_axis_shift
// CHECK:         vector.broadcast %{{.*}} : vector<1x4x1xf32> to vector<1x4x4xf32>
// CHECK-NEXT:    vector.shape_cast %{{.*}} : vector<1x4x4xf32> to vector<4x4x1xf32>
func.func @no_fold_bcast_axis_shift(%arg0: vector<1x4x1xf32>) -> vector<4x4x1xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4x1xf32> to vector<1x4x4xf32>
  %1 = vector.shape_cast %0 : vector<1x4x4xf32> to vector<4x4x1xf32>
  return %1 : vector<4x4x1xf32>
}

// -----

// CHECK-LABEL: @fold_bcast_leading_dims
// CHECK:         %[[RES:.*]] = vector.broadcast %{{.*}} : vector<3xf32> to vector<8x3xf32>
// CHECK-NEXT:    return %[[RES]] : vector<8x3xf32>
func.func @fold_bcast_leading_dims(%arg0: vector<3xf32>) -> vector<8x3xf32> {
  %0 = vector.broadcast %arg0 : vector<3xf32> to vector<2x4x3xf32>
  %1 = vector.shape_cast %0 : vector<2x4x3xf32> to vector<8x3xf32>
  return %1 : vector<8x3xf32>
}

// -----

// CHECK-LABEL: @fold_bcast_src_leading_unit_dims
// CHECK:         %[[RES:.*]] = vector.broadcast %{{.*}} : vector<1x3xf32> to vector<4x3xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4x3xf32>
func.func @fold_bcast_src_leading_unit_dims(%arg0: vector<1x3xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<1x3xf32> to vector<4x1x3xf32>
  %1 = vector.shape_cast %0 : vector<4x1x3xf32> to vector<4x3xf32>
  return %1 : vector<4x3xf32>
}

// -----

// CHECK-LABEL: @fold_bcast_consecutive_unit_dims
// CHECK:         %[[RES:.*]] = vector.broadcast %{{.*}} : vector<2x1x1x3xf32> to vector<2x2x2x3xf32>
// CHECK-NEXT:    return %[[RES]] : vector<2x2x2x3xf32>
func.func @fold_bcast_consecutive_unit_dims(%arg0: vector<2x1x1x3xf32>) -> vector<2x2x2x3xf32> {
  %0 = vector.broadcast %arg0 : vector<2x1x1x3xf32> to vector<2x1x4x3xf32>
  %1 = vector.shape_cast %0 : vector<2x1x4x3xf32> to vector<2x2x2x3xf32>
  return %1 : vector<2x2x2x3xf32>
}

// -----

// CHECK-LABEL: @no_fold_bcast_scalable_vs_fixed
// CHECK:         vector.broadcast %{{.*}} : vector<2x1x1xf32> to vector<2x[4]x2xf32>
// CHECK-NEXT:    vector.shape_cast %{{.*}} : vector<2x[4]x2xf32> to vector<[1]x2x4x2xf32>
func.func @no_fold_bcast_scalable_vs_fixed(%arg0: vector<2x1x1xf32>) -> vector<[1]x2x4x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2x1x1xf32> to vector<2x[4]x2xf32>
  %1 = vector.shape_cast %0 : vector<2x[4]x2xf32> to vector<[1]x2x4x2xf32>
  return %1 : vector<[1]x2x4x2xf32>
}

// -----

// CHECK-LABEL: @no_fold_bcast_scalable_axis_shift
// CHECK:         vector.broadcast %{{.*}} : vector<1x[4]x1xf32> to vector<1x[4]x4xf32>
// CHECK-NEXT:    vector.shape_cast %{{.*}} : vector<1x[4]x4xf32> to vector<4x[4]x1xf32>
func.func @no_fold_bcast_scalable_axis_shift(%arg0: vector<1x[4]x1xf32>) -> vector<4x[4]x1xf32> {
  %0 = vector.broadcast %arg0 : vector<1x[4]x1xf32> to vector<1x[4]x4xf32>
  %1 = vector.shape_cast %0 : vector<1x[4]x4xf32> to vector<4x[4]x1xf32>
  return %1 : vector<4x[4]x1xf32>
}

// -----

// CHECK-LABEL: @fold_bcast_leading_scalable_dims
// CHECK:         %[[RES:.*]] = vector.broadcast %{{.*}} : vector<[4]xf32> to vector<8x[4]xf32>
// CHECK-NEXT:    return %[[RES]] : vector<8x[4]xf32>
func.func @fold_bcast_leading_scalable_dims(%arg0: vector<[4]xf32>) -> vector<8x[4]xf32> {
  %0 = vector.broadcast %arg0 : vector<[4]xf32> to vector<2x4x[4]xf32>
  %1 = vector.shape_cast %0 : vector<2x4x[4]xf32> to vector<8x[4]xf32>
  return %1 : vector<8x[4]xf32>
}

// -----

// CHECK-LABEL: @fold_bcast_leading_scalable_dim_fixed_src
// CHECK:         %[[RES:.*]] = vector.broadcast %{{.*}} : vector<4xf32> to vector<[8]x4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<[8]x4xf32>
func.func @fold_bcast_leading_scalable_dim_fixed_src(%arg0: vector<4xf32>) -> vector<[8]x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<2x[4]x4xf32>
  %1 = vector.shape_cast %0 : vector<2x[4]x4xf32> to vector<[8]x4xf32>
  return %1 : vector<[8]x4xf32>
}

// -----

// CHECK-LABEL: @fold_bcast_scalable_consecutive_unit_dims
// CHECK:         %[[RES:.*]] = vector.broadcast %{{.*}} : vector<[2]x1x1x3xf32> to vector<[2]x2x2x3xf32>
// CHECK-NEXT:    return %[[RES]] : vector<[2]x2x2x3xf32>
func.func @fold_bcast_scalable_consecutive_unit_dims(%arg0: vector<[2]x1x1x3xf32>) -> vector<[2]x2x2x3xf32> {
  %0 = vector.broadcast %arg0 : vector<[2]x1x1x3xf32> to vector<[2]x1x4x3xf32>
  %1 = vector.shape_cast %0 : vector<[2]x1x4x3xf32> to vector<[2]x2x2x3xf32>
  return %1 : vector<[2]x2x2x3xf32>
}

// -----

// CHECK-LABEL: @no_fold_bcast_scalable_unit_dim_axis_shift
// CHECK:         vector.broadcast %{{.*}} : vector<[1]x1xf32> to vector<[1]x4xf32>
// CHECK-NEXT:    vector.shape_cast %{{.*}} : vector<[1]x4xf32> to vector<4x[1]xf32>
func.func @no_fold_bcast_scalable_unit_dim_axis_shift(%arg0: vector<[1]x1xf32>) -> vector<4x[1]xf32> {
  %0 = vector.broadcast %arg0 : vector<[1]x1xf32> to vector<[1]x4xf32>
  %1 = vector.shape_cast %0 : vector<[1]x4xf32> to vector<4x[1]xf32>
  return %1 : vector<4x[1]xf32>
}

// -----

// CHECK-LABEL: @fold_bcast_scalar
// CHECK:         %[[RES:.*]] = vector.broadcast %{{.*}} : f32 to vector<2x4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<2x4xf32>
func.func @fold_bcast_scalar(%arg0: f32) -> vector<2x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<8xf32>
  %1 = vector.shape_cast %0 : vector<8xf32> to vector<2x4xf32>
  return %1 : vector<2x4xf32>
}

// -----

// CHECK-LABEL: @fold_bcast_scalar_scalable
// CHECK:         %[[RES:.*]] = vector.broadcast %{{.*}} : f32 to vector<2x[4]xf32>
// CHECK-NEXT:    return %[[RES]] : vector<2x[4]xf32>
func.func @fold_bcast_scalar_scalable(%arg0: f32) -> vector<2x[4]xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<[8]xf32>
  %1 = vector.shape_cast %0 : vector<[8]xf32> to vector<2x[4]xf32>
  return %1 : vector<2x[4]xf32>
}
