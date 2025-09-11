// RUN: mlir-opt %s -test-unroll-vector-to-elements -split-input-file | FileCheck %s
// RUN: mlir-opt %s -transform-preload-library='transform-library-paths=%p/td/unroll-elements.mlir' \
// RUN: -transform-interpreter=entry-point=unroll_to_elements | FileCheck %s

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
// CHECK:         %[[VEC0:.+]] = vector.extract %[[ARG0]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK:         %[[VEC1:.+]] = vector.extract %[[ARG0]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK:         %[[RES0:.+]]:2 = vector.to_elements %[[VEC0]] : vector<2xf32>
// CHECK:         %[[RES1:.+]]:2 = vector.to_elements %[[VEC1]] : vector<2xf32>
// CHECK:         return %[[RES0]]#0, %[[RES0]]#1, %[[RES1]]#0, %[[RES1]]#1
func.func @to_elements_2d(%arg0: vector<2x2xf32>) -> (f32, f32, f32, f32) {
  %0:4 = vector.to_elements %arg0 : vector<2x2xf32>
  return %0#0, %0#1, %0#2, %0#3 : f32, f32, f32, f32
}
