// RUN: mlir-opt %s -split-input-file -canonicalize |  FileCheck %s


// +----------------------------------------
//  Tests of BroadcastToShapeCast
// +----------------------------------------

// CHECK-LABEL: @broadcast_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<4xi8>
//  CHECK-NEXT: %[[SCAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT: return %[[SCAST]] : vector<1x1x4xi8>
func.func @broadcast_to_shape_cast(%arg0 : vector<4xi8>) -> vector<1x1x4xi8> {
  %0 = vector.broadcast %arg0 : vector<4xi8> to vector<1x1x4xi8>
  return %0 : vector<1x1x4xi8>
}

// -----

// broadcast can only be transformed to a shape_cast if the number of elements is
// unchanged by the broadcast
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

// +----------------------------------------
//  Tests of TransposeToShapeCast
// +----------------------------------------

// In this test, the permutation maps the non-unit dimensions (0 and 2) as follows:
// 0 -> 0
// 2 -> 1
// Because 0 < 1, this permutation is order preserving and effectively a shape_cast.
// CHECK-LABEL: @transpose_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<2x1x2xf32>
//  CHECK-NEXT: %[[SCAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT: return %[[SCAST]] : vector<2x2x1xf32>
func.func @transpose_to_shape_cast(%arg0 : vector<2x1x2xf32>) -> vector<2x2x1xf32> {
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<2x1x2xf32> to vector<2x2x1xf32>
  return %0 : vector<2x2x1xf32>
}

// -----

// In this test, the permutation maps the non-unit dimensions (1 and 2) as follows:
// 1 -> 0
// 2 -> 4
// Because 0 < 4, this permutation is order preserving and effectively a shape_cast.
// CHECK-LABEL: @shape_cast_of_transpose
//  CHECK-SAME: %[[ARG:.*]]: vector<1x4x4x1x1xi8>)
//       CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG]] :
//  CHECK-SAME: vector<1x4x4x1x1xi8> to vector<4x1x1x1x4xi8>
//       CHECK: return %[[SHAPE_CAST]]
func.func @shape_cast_of_transpose(%arg : vector<1x4x4x1x1xi8>) -> vector<4x1x1x1x4xi8> {
  %0 = vector.transpose %arg, [1, 0, 3, 4, 2]  : vector<1x4x4x1x1xi8> to vector<4x1x1x1x4xi8>
  return %0 : vector<4x1x1x1x4xi8>
}

// -----

// Scalable dimensions should be treated as non-unit dimensions.
// CHECK-LABEL: @transpose_scalable_unit
//   CHECK-NOT: shape_cast
func.func @transpose_scalable_unit(%arg : vector<[1]x4xi8>) -> vector<4x[1]xi8> {
  %0 = vector.transpose %arg, [1, 0] : vector<[1]x4xi8> to vector<4x[1]xi8>
  return %0 : vector<4x[1]xi8>
}

// -----

// In this test, the mapping of non-unit dimensions (1 and 2) is as follows:
// 1 -> 2
// 2 -> 1
// As this is not increasing (2 > 1), this transpose is not order
// preserving and cannot be treated as a shape_cast.
// CHECK-LABEL: @negative_transpose_to_shape_cast
//   CHECK-NOT: shape_cast
func.func @negative_transpose_to_shape_cast(%arg : vector<1x4x4x1xi8>) -> vector<1x4x4x1xi8> {
  %0 = vector.transpose %arg, [0, 2, 1, 3]
     : vector<1x4x4x1xi8> to vector<1x4x4x1xi8>
  return %0 : vector<1x4x4x1xi8>
}

// -----

// CHECK-LABEL: @shape_cast_of_transpose_scalable
//  CHECK-NEXT: vector.shape_cast
//  CHECK-NEXT: return
func.func @shape_cast_of_transpose_scalable(%arg : vector<[4]x1xi8>) -> vector<[4]xi8> {
  %0 = vector.transpose %arg, [1, 0] : vector<[4]x1xi8> to vector<1x[4]xi8>
  %1 = vector.shape_cast %0 : vector<1x[4]xi8> to vector<[4]xi8>
  return %1 : vector<[4]xi8>
}

// -----

// CHECK-LABEL: @transpose_of_shape_cast_scalable
//  CHECK-NEXT: vector.shape_cast
//  CHECK-NEXT: return
func.func @transpose_of_shape_cast_scalable(%arg : vector<[4]xi8>) -> vector<[4]x1xi8> {
  %0 = vector.shape_cast %arg : vector<[4]xi8> to vector<1x[4]xi8>
  %1 = vector.transpose %0, [1, 0] : vector<1x[4]xi8> to vector<[4]x1xi8>
  return %1 : vector<[4]x1xi8>
}

// -----

// A test where a transpose cannot be transformed to a shape_cast because it is not order
// preserving
// CHECK-LABEL: @negative_transpose_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<2x1x2xf32>
//  CHECK-NEXT: %[[TRANSPOSE:.*]] = vector.transpose %[[ARG0]], [2, 0, 1]
//  CHECK-NEXT: return %[[TRANSPOSE]] : vector<2x2x1xf32>
func.func @negative_transpose_to_shape_cast(%arg0 : vector<2x1x2xf32>) -> vector<2x2x1xf32> {
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<2x1x2xf32> to vector<2x2x1xf32>
  return %0 : vector<2x2x1xf32>
}

// -----

// +----------------------------------------
//  Tests of ExtractToShapeCast
// +----------------------------------------

// CHECK-LABEL: @extract_to_shape_cast
//  CHECK-SAME: %[[ARG0:.*]]: vector<1x4xf32>
//  CHECK-NEXT: %[[SCAST:.*]] = vector.shape_cast %[[ARG0]]
//  CHECK-NEXT: return %[[SCAST]] : vector<4xf32>
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

