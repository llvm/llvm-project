// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file -allow-unregistered-dialect | FileCheck %s

// This file contains some canonicalizations tests involving vector.transpose.

// CHECK-LABEL: func @transpose_scalar_broadcast1
//  CHECK-SAME: (%[[ARG:.+]]: vector<1xf32>)
//       CHECK:   %[[V:.+]] = vector.broadcast %[[ARG]] : vector<1xf32> to vector<1x8xf32>
//       CHECK:   return %[[V]] : vector<1x8xf32>
func.func @transpose_scalar_broadcast1(%value: vector<1xf32>) -> vector<1x8xf32> {
  %bcast = vector.broadcast %value : vector<1xf32> to vector<8x1xf32>
  %t = vector.transpose %bcast, [1, 0] : vector<8x1xf32> to vector<1x8xf32>
  return %t : vector<1x8xf32>
}

// -----

// CHECK-LABEL: func @transpose_scalar_broadcast2
//  CHECK-SAME: (%[[ARG:.+]]: f32)
//       CHECK:   %[[V:.+]] = vector.broadcast %[[ARG]] : f32 to vector<1x8xf32>
//       CHECK:   return %[[V]] : vector<1x8xf32>
func.func @transpose_scalar_broadcast2(%value: f32) -> vector<1x8xf32> {
  %bcast = vector.broadcast %value : f32 to vector<8x1xf32>
  %t = vector.transpose %bcast, [1, 0] : vector<8x1xf32> to vector<1x8xf32>
  return %t : vector<1x8xf32>
}

// -----


// CHECK-LABEL: broadcast_transpose_scalar_to_broadcast
//  CHECK-SAME:  %[[ARG:.*]]: i8) -> vector<2x3x4xi8> {
func.func @broadcast_transpose_scalar_to_broadcast(%arg0 : i8) -> vector<2x3x4xi8> {
//       CHECK:  %[[BC:.*]] = vector.broadcast %[[ARG]] : i8 to vector<2x3x4xi8>
  %0 = vector.broadcast %arg0 : i8 to vector<3x4x2xi8>
  %1 = vector.transpose %0, [2, 0, 1] : vector<3x4x2xi8> to vector<2x3x4xi8>
//       CHECK:  return %[[BC]] : vector<2x3x4xi8>
  return %1 : vector<2x3x4xi8>
}

// -----

// CHECK-LABEL: broadcast_transpose_ones_to_broadcast
//  CHECK-SAME:  %[[ARG:.*]]: vector<1x1x1xi8>) -> vector<2x3x4xi8> {
//       CHECK:  %[[RES:.*]] = vector.broadcast %[[ARG]] : vector<1x1x1xi8> to vector<2x3x4xi8>
//       CHECK:  return %[[RES]] : vector<2x3x4xi8>
func.func @broadcast_transpose_ones_to_broadcast(%arg0 : vector<1x1x1xi8>) -> vector<2x3x4xi8> {
  %0 = vector.broadcast %arg0 : vector<1x1x1xi8> to vector<3x4x2xi8>
  %1 = vector.transpose %0, [2, 0, 1] : vector<3x4x2xi8> to vector<2x3x4xi8>
  return %1 : vector<2x3x4xi8>
}

// -----

// CHECK-LABEL: broadcast_transpose_partial_ones_to_broadcast
//  CHECK-SAME:  %[[ARG:.*]]: vector<1xi8>) -> vector<8x1xi8> {
//       CHECK:  %[[RES:.*]] = vector.broadcast %[[ARG]] : vector<1xi8> to vector<8x1xi8>
//       CHECK:  return %[[RES]] : vector<8x1xi8>
func.func @broadcast_transpose_partial_ones_to_broadcast(%arg0 : vector<1xi8>) -> vector<8x1xi8> {
  %0 = vector.broadcast %arg0 : vector<1xi8> to vector<1x8xi8>
  %1 = vector.transpose %0, [1, 0] : vector<1x8xi8> to vector<8x1xi8>
  return %1 : vector<8x1xi8>
}

// -----

// CHECK-LABEL: broadcast_transpose_mixed_example
//  CHECK-SAME:  %[[ARG:.*]]: vector<4x1x1x7xi8>) -> vector<3x2x4x5x6x7xi8> {
//       CHECK:  %[[RES:.*]] = vector.broadcast %[[ARG]] : vector<4x1x1x7xi8> to vector<3x2x4x5x6x7xi8>
//       CHECK:  return %[[RES]] : vector<3x2x4x5x6x7xi8>
func.func @broadcast_transpose_mixed_example(%arg0 : vector<4x1x1x7xi8>) -> vector<3x2x4x5x6x7xi8> {
  %0 = vector.broadcast %arg0 : vector<4x1x1x7xi8> to vector<2x3x4x5x6x7xi8>
  %1 = vector.transpose %0, [1, 0, 2, 3, 4, 5] : vector<2x3x4x5x6x7xi8> to vector<3x2x4x5x6x7xi8>
  return %1 : vector<3x2x4x5x6x7xi8>
}

// -----

// CHECK-LABEL: broadcast_transpose_final_group
//  CHECK-SAME:  %[[ARG:.*]]: vector<4x7x1x1xi8>) -> vector<4x7x2x3xi8> {
//       CHECK:  %[[RES:.*]] = vector.broadcast %[[ARG]] : vector<4x7x1x1xi8> to vector<4x7x2x3xi8>
//       CHECK:  return %[[RES]] : vector<4x7x2x3xi8>
func.func @broadcast_transpose_final_group(%arg0 : vector<4x7x1x1xi8>) -> vector<4x7x2x3xi8> {
  %0 = vector.broadcast %arg0 : vector<4x7x1x1xi8> to vector<4x7x3x2xi8>
  %1 = vector.transpose %0, [0, 1, 3, 2] : vector<4x7x3x2xi8> to vector<4x7x2x3xi8>
  return %1 : vector<4x7x2x3xi8>
}

// -----

// CHECK-LABEL: negative_broadcast_transpose_square
//  CHECK-SAME:  %[[ARG:.*]]:
//       CHECK:  %[[BCT:.*]] = vector.broadcast %[[ARG]]
//       CHECK:  %[[TRP:.*]] = vector.transpose %[[BCT]], [1, 0]
//       CHECK:  return %[[TRP]] : vector<4x4xi8>
func.func @negative_broadcast_transpose_square(%arg0 : vector<4x1xi8>) -> vector<4x4xi8> {
  %0 = vector.broadcast %arg0 : vector<4x1xi8> to vector<4x4xi8>
  %1 = vector.transpose %0, [1, 0] : vector<4x4xi8> to vector<4x4xi8>
  return %1 : vector<4x4xi8>
}

// -----

// CHECK-LABEL: negative_broadcast_transpose_hypercube
//  CHECK-SAME:  %[[ARG:.*]]:
//       CHECK:  %[[BCT:.*]] = vector.broadcast %[[ARG]]
//       CHECK:  %[[TRP:.*]] = vector.transpose %[[BCT]], [1, 0, 3, 2]
//       CHECK:  return %[[TRP]] : vector<4x4x4x4xi8>
func.func @negative_broadcast_transpose_hypercube(%arg0 : vector<1x1x4xi8>) -> vector<4x4x4x4xi8> {
  %0 = vector.broadcast %arg0 : vector<1x1x4xi8> to vector<4x4x4x4xi8>
  %1 = vector.transpose %0, [1, 0, 3, 2] : vector<4x4x4x4xi8> to vector<4x4x4x4xi8>
  return %1 : vector<4x4x4x4xi8>
}

// -----

// CHECK-LABEL: negative_broadcast_transpose_102
//  CHECK-SAME:  %[[ARG:.*]]:
//       CHECK:  %[[BCT:.*]] = vector.broadcast %[[ARG]]
//       CHECK:  %[[TRP:.*]] = vector.transpose %[[BCT]], [1, 0, 2]
//       CHECK:  return %[[TRP]] : vector<3x3x3xi8>
func.func @negative_broadcast_transpose_102(%arg0 : vector<3x1x3xi8>) -> vector<3x3x3xi8> {
  %0 = vector.broadcast %arg0 : vector<3x1x3xi8> to vector<3x3x3xi8>
  %1 = vector.transpose %0, [1, 0, 2] : vector<3x3x3xi8> to vector<3x3x3xi8>
  return %1 : vector<3x3x3xi8>
}

// -----

// CHECK-LABEL: negative_broadcast_transpose_021
//  CHECK-SAME:  %[[ARG:.*]]:
//       CHECK:  %[[BCT:.*]] = vector.broadcast %[[ARG]]
//       CHECK:  %[[TRP:.*]] = vector.transpose %[[BCT]], [0, 2, 1]
//       CHECK:  return %[[TRP]] : vector<3x3x3xi8>
func.func @negative_broadcast_transpose_021(%arg0 : vector<3x1x3xi8>) -> vector<3x3x3xi8> {
  %0 = vector.broadcast %arg0 : vector<3x1x3xi8> to vector<3x3x3xi8>
  %1 = vector.transpose %0, [0, 2, 1] : vector<3x3x3xi8> to vector<3x3x3xi8>
  return %1 : vector<3x3x3xi8>
}


// -----

// Test of FoldTransposeShapeCast
// In this test, the permutation maps the non-unit dimensions (1 and 2) as follows:
// 1 -> 0
// 2 -> 4
// Because 0 < 4, this permutation is order preserving and effectively a shape_cast.
// CHECK-LABEL: @transpose_shape_cast
//  CHECK-SAME:   %[[ARG:.*]]: vector<1x4x4x1x1xi8>) -> vector<4x4xi8> {
//       CHECK:   %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG]] :
//  CHECK-SAME:   vector<1x4x4x1x1xi8> to vector<4x4xi8>
//       CHECK:   return %[[SHAPE_CAST]] : vector<4x4xi8>
func.func @transpose_shape_cast(%arg : vector<1x4x4x1x1xi8>) -> vector<4x4xi8> {
  %0 = vector.transpose %arg, [1, 0, 3, 4, 2]
     : vector<1x4x4x1x1xi8> to vector<4x1x1x1x4xi8>
  %1 = vector.shape_cast %0 : vector<4x1x1x1x4xi8> to vector<4x4xi8>
  return %1 : vector<4x4xi8>
}

// -----

// Test of FoldTransposeShapeCast
// In this test, the mapping of non-unit dimensions (1 and 2) is as follows:
// 1 -> 2
// 2 -> 1
// As this is not increasing (2 > 1), this transpose is not order
// preserving and cannot be treated as a shape_cast.
// CHECK-LABEL: @negative_transpose_shape_cast
//  CHECK-SAME:   %[[ARG:.*]]: vector<1x4x4x1xi8>) -> vector<4x4xi8> {
//       CHECK:   %[[TRANSPOSE:.*]] = vector.transpose %[[ARG]]
//       CHECK:   %[[SHAPE_CAST:.*]] = vector.shape_cast %[[TRANSPOSE]]
//       CHECK:   return %[[SHAPE_CAST]] : vector<4x4xi8>
func.func @negative_transpose_shape_cast(%arg : vector<1x4x4x1xi8>) -> vector<4x4xi8> {
  %0 = vector.transpose %arg, [0, 2, 1, 3]
     : vector<1x4x4x1xi8> to vector<1x4x4x1xi8>
  %1 = vector.shape_cast %0 : vector<1x4x4x1xi8> to vector<4x4xi8>
  return %1 : vector<4x4xi8>
}

// -----

// Test of FoldTransposeShapeCast
// Currently the conversion shape_cast(transpose) -> shape_cast is disabled for
// scalable vectors because of bad interaction with ConvertIllegalShapeCastOpsToTransposes
// CHECK-LABEL: @negative_transpose_shape_cast_scalable
//       CHECK:  vector.transpose
//       CHECK:  vector.shape_cast
func.func @negative_transpose_shape_cast_scalable(%arg : vector<[4]x1xi8>) -> vector<[4]xi8> {
  %0 = vector.transpose %arg, [1, 0] : vector<[4]x1xi8> to vector<1x[4]xi8>
  %1 = vector.shape_cast %0 : vector<1x[4]xi8> to vector<[4]xi8>
  return %1 : vector<[4]xi8>
}

// -----

// Test of shape_cast folding.
// The conversion transpose(shape_cast) -> shape_cast is not disabled for scalable
// vectors.
// CHECK-LABEL: @shape_cast_transpose_scalable
//       CHECK: vector.shape_cast
//  CHECK-SAME: vector<[4]xi8> to vector<[4]x1xi8>
func.func @shape_cast_transpose_scalable(%arg : vector<[4]xi8>) -> vector<[4]x1xi8> {
  %0 = vector.shape_cast %arg : vector<[4]xi8> to vector<1x[4]xi8>
  %1 = vector.transpose %0, [1, 0] : vector<1x[4]xi8> to vector<[4]x1xi8>
  return %1 : vector<[4]x1xi8>
}

// -----

// Test of shape_cast folding.
// A transpose that is 'order preserving' can be treated like a shape_cast. 
// CHECK-LABEL: @shape_cast_transpose
//  CHECK-SAME:   %[[ARG:.*]]: vector<2x3x1x1xi8>) -> vector<6x1x1xi8> {
//       CHECK:   %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG]] :
//  CHECK-SAME:   vector<2x3x1x1xi8> to vector<6x1x1xi8>
//       CHECK:   return %[[SHAPE_CAST]] : vector<6x1x1xi8>
func.func @shape_cast_transpose(%arg : vector<2x3x1x1xi8>) ->  vector<6x1x1xi8> {
  %0 = vector.shape_cast %arg : vector<2x3x1x1xi8> to vector<6x1x1xi8>
  %1 = vector.transpose %0, [0, 2, 1]
     : vector<6x1x1xi8> to vector<6x1x1xi8>
  return %1 : vector<6x1x1xi8>
}

// -----

// Test of shape_cast folding.
// Scalable dimensions should be treated as non-unit dimensions.
// CHECK-LABEL: @shape_cast_transpose_scalable
//       CHECK: vector.shape_cast
//       CHECK: vector.transpose
func.func @shape_cast_transpose_scalable_unit(%arg : vector<[1]x4x1xi8>) -> vector<4x[1]xi8> {
  %0 = vector.shape_cast %arg : vector<[1]x4x1xi8> to vector<[1]x4xi8>
  %1 = vector.transpose %0, [1, 0] : vector<[1]x4xi8> to vector<4x[1]xi8>
  return %1 : vector<4x[1]xi8>
}

// -----

// Test of shape_cast (not) folding.
// CHECK-LABEL: @negative_shape_cast_transpose
//  CHECK-SAME:   %[[ARG:.*]]: vector<6xi8>) -> vector<2x3xi8> {
//       CHECK:   %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG]] :
//       CHECK:   %[[TRANSPOSE:.*]] = vector.transpose %[[SHAPE_CAST]]
//       CHECK:   return %[[TRANSPOSE]] : vector<2x3xi8>
func.func @negative_shape_cast_transpose(%arg : vector<6xi8>) -> vector<2x3xi8> {
  %0 = vector.shape_cast %arg : vector<6xi8> to vector<3x2xi8>
  %1 = vector.transpose %0, [1, 0] : vector<3x2xi8> to vector<2x3xi8>
  return %1 : vector<2x3xi8>
}
