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

