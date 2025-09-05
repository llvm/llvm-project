// RUN: mlir-opt %s -test-flatten-vector-to-elements -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @to_elements_1d(
// CHECK-SAME:    %[[ARG0:.+]]: vector<2xf32>
// CHECK:         %[[RES:.+]]:2 = vector.to_elements %[[ARG0]] : vector<2xf32>
// CHECK:         return %[[RES]]#0, %[[RES]]#1
func.func @to_elements_1d(%arg0: vector<2xf32>) -> (f32, f32) {
  %0:2 = vector.to_elements %arg0 : vector<2xf32>
  return %0#0, %0#1 : f32, f32
}

// -----

// CHECK-LABEL: func.func @to_elements_2d(
// CHECK-SAME:    %[[ARG0:.+]]: vector<2x2xf32>
// CHECK:         %[[CAST:.+]] = vector.shape_cast %[[ARG0]]
// CHECK:         %[[RES:.+]]:4 = vector.to_elements %[[CAST]] : vector<4xf32>
// CHECK:         return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2, %[[RES]]#3
func.func @to_elements_2d(%arg0: vector<2x2xf32>) -> (f32, f32, f32, f32) {
  %0:4 = vector.to_elements %arg0 : vector<2x2xf32>
  return %0#0, %0#1, %0#2, %0#3 : f32, f32, f32, f32
}
