// RUN: mlir-opt %s -test-vector-to-vector-lowering | FileCheck %s

// CHECK-LABEL: broadcast_to_shapecast
//       CHECK:   %[[C:.*]] = vector.shape_cast %{{.*}} : vector<4x4xf16> to vector<1x4x4xf16>
//  CHECK-NEXT:   return %[[C]] : vector<1x4x4xf16>
func @broadcast_to_shapecast(%arg0: vector<4x4xf16>) -> vector<1x4x4xf16> {
  %0 = vector.broadcast %arg0 : vector<4x4xf16> to vector<1x4x4xf16>
  return %0 : vector<1x4x4xf16>
}

// -----

// CHECK-LABEL: func @insert_extract_to_shapecast
//  CHECK-SAME: (%[[ARG0:.*]]: vector<1x1x4xf32>, %[[ARG1:.*]]: vector<4xf32>)
//       CHECK:   %[[V0:.*]] = vector.shape_cast %[[ARG0]] : vector<1x1x4xf32> to vector<4xf32>
//       CHECK:   %[[V1:.*]] = vector.shape_cast %[[ARG1]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return %[[V0]], %[[V1]] : vector<4xf32>, vector<1x1x4xf32>
func @insert_extract_to_shapecast(%arg0 : vector<1x1x4xf32>,
  %arg1 : vector<4xf32>) -> (vector<4xf32>, vector<1x1x4xf32>) {
  %0 = vector.extract %arg0[0, 0] : vector<1x1x4xf32>
  %1 = vector.insert %arg1, %arg0 [0, 0] : vector<4xf32> into vector<1x1x4xf32>
  return %0, %1 : vector<4xf32>, vector<1x1x4xf32>
}
