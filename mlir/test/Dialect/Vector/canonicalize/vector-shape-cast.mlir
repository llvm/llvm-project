// RUN: mlir-opt %s -split-input-file -canonicalize |  FileCheck %s

// +----------------------------------------
//  Tests of TransposeToShapeCast
// +----------------------------------------

// CHECK-LABEL: @transpose_to_shape_cast
//  CHECK-SAME:  %[[ARG0:.*]]: vector<2x1x2xf32>
//  CHECK-NEXT:  %[[SCAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT:  return %[[SCAST]] : vector<2x2x1xf32>
func.func @transpose_to_shape_cast(%arg0 : vector<2x1x2xf32>) -> vector<2x2x1xf32> {
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<2x1x2xf32> to vector<2x2x1xf32>
  return %0 : vector<2x2x1xf32>
}


// -----

// CHECK-LABEL: @negative_transpose_to_shape_cast
//  CHECK-SAME:  %[[ARG0:.*]]: vector<2x1x2xf32>
//  CHECK-NEXT:  %[[TRANSPOSE:.*]] = vector.transpose %[[ARG0]], [2, 0, 1]
//  CHECK-NEXT:  return %[[TRANSPOSE]] : vector<2x2x1xf32>
func.func @negative_transpose_to_shape_cast(%arg0 : vector<2x1x2xf32>) -> vector<2x2x1xf32> {
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<2x1x2xf32> to vector<2x2x1xf32>
  return %0 : vector<2x2x1xf32>
}

// -----

// +----------------------------------------
//  Tests of BroadcastToShapeCast
// +----------------------------------------

// CHECK-LABEL: @broadcast_to_shape_cast
//  CHECK-SAME:  %[[ARG0:.*]]: vector<4xi8>
//  CHECK-NEXT:  %[[SCAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT:  return %[[SCAST]] : vector<1x1x4xi8>
func.func @broadcast_to_shape_cast(%arg0 : vector<4xi8>) -> vector<1x1x4xi8> {
  %0 = vector.broadcast %arg0 : vector<4xi8> to vector<1x1x4xi8>
  return %0 : vector<1x1x4xi8>
}

// -----

// CHECK-LABEL: @negative_broadcast_to_shape_cast
//   CHECK-NOT: shape_cast
//       CHECK: return
func.func @negative_broadcast_to_shape_cast(%arg0 : vector<1x4xi8>) -> vector<2x3x4xi8> {
  %0 = vector.broadcast %arg0 : vector<1x4xi8> to vector<2x3x4xi8>
  return %0 : vector<2x3x4xi8>
}

// -----

// CHECK-LABEL: @negative_broadcast_scalar_to_shape_cast
//   CHECK-NOT: shape_cast
//       CHECK: return
func.func @negative_broadcast_scalar_to_shape_cast(%arg0 : i8) -> vector<1xi8> {
  %0 = vector.broadcast %arg0 : i8 to vector<1xi8>
  return %0 : vector<1xi8>
}

// -----

// The conversion transpose(shape_cast) -> shape_cast is currently disabled for scalable
// vectors.
// CHECK-LABEL: @transpose_of_shape_cast_scalable
//       CHECK: vector.shape_cast
//       CHECK: vector.transpose
func.func @transpose_of_shape_cast_scalable(%arg : vector<[4]xi8>) -> vector<[4]x1xi8> {
  %0 = vector.shape_cast %arg : vector<[4]xi8> to vector<1x[4]xi8>
  %1 = vector.transpose %0, [1, 0] : vector<1x[4]xi8> to vector<[4]x1xi8>
  return %1 : vector<[4]x1xi8>
}

// -----

// A transpose that is 'order preserving' can be treated like a shape_cast.
// CHECK-LABEL: @transpose_of_shape_cast
//  CHECK-SAME:   %[[ARG:.*]]: vector<2x3x1x1xi8>) -> vector<6x1x1xi8> {
//       CHECK:   %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG]] :
//  CHECK-SAME:   vector<2x3x1x1xi8> to vector<6x1x1xi8>
//       CHECK:   return %[[SHAPE_CAST]] : vector<6x1x1xi8>
func.func @transpose_of_shape_cast(%arg : vector<2x3x1x1xi8>) ->  vector<6x1x1xi8> {
  %0 = vector.shape_cast %arg : vector<2x3x1x1xi8> to vector<6x1x1xi8>
  %1 = vector.transpose %0, [0, 2, 1]
     : vector<6x1x1xi8> to vector<6x1x1xi8>
  return %1 : vector<6x1x1xi8>
}

// -----

// Scalable dimensions should be treated as non-unit dimensions.
// CHECK-LABEL: @transpose_of_shape_cast_scalable
//       CHECK: vector.shape_cast
//       CHECK: vector.transpose
func.func @transpose_of_shape_cast_scalable_unit(%arg : vector<[1]x4x1xi8>) -> vector<4x[1]xi8> {
  %0 = vector.shape_cast %arg : vector<[1]x4x1xi8> to vector<[1]x4xi8>
  %1 = vector.transpose %0, [1, 0] : vector<[1]x4xi8> to vector<4x[1]xi8>
  return %1 : vector<4x[1]xi8>
}

// -----

// Test of shape_cast (not) folding.
// CHECK-LABEL: @negative_transpose_of_shape_cast
//  CHECK-SAME:   %[[ARG:.*]]: vector<6xi8>) -> vector<2x3xi8> {
//       CHECK:   %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG]] :
//       CHECK:   %[[TRANSPOSE:.*]] = vector.transpose %[[SHAPE_CAST]]
//       CHECK:   return %[[TRANSPOSE]] : vector<2x3xi8>
func.func @negative_transpose_of_shape_cast(%arg : vector<6xi8>) -> vector<2x3xi8> {
  %0 = vector.shape_cast %arg : vector<6xi8> to vector<3x2xi8>
  %1 = vector.transpose %0, [1, 0] : vector<3x2xi8> to vector<2x3xi8>
  return %1 : vector<2x3xi8>
}

// -----

// +----------------------------------------
//  Tests of ExtractToShapeCast
// +----------------------------------------

// CHECK-LABEL: @extract_to_shape_cast
//  CHECK-SAME:  %[[ARG0:.*]]: vector<1x4xf32>
//  CHECK-NEXT:  %[[SCAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT:  return %[[SCAST]] : vector<4xf32>
func.func @extract_to_shape_cast(%arg0 : vector<1x4xf32>) -> vector<4xf32> {
  %0 = vector.extract %arg0[0] : vector<4xf32> from vector<1x4xf32>
  return %0 : vector<4xf32>
}

// -----

// In this example, arg1 might be negative indicating poison.
// CHECK-LABEL: @negative_extract_to_shape_cast
//   CHECK-NOT: shape_cast
func.func @negative_extract_to_shape_cast(%arg0 : vector<1x4xf32>, %arg1 : index) -> vector<4xf32> {
  %0 = vector.extract %arg0[%arg1] : vector<4xf32> from vector<1x4xf32>
  return %0 : vector<4xf32>
}

