// RUN: mlir-opt %s -split-input-file -test-convert-to-shape-cast |  FileCheck %s 


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

