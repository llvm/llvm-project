// RUN: mlir-opt %s -transform-preload-library='transform-library-paths=%p/td/unroll-elements.mlir' \
// RUN: -transform-interpreter=entry-point=unroll_to_elements | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollToElements.
//===----------------------------------------------------------------------===//

// 1-D vector.from_elements should not be unrolled.

// CHECK-LABEL: func.func @negative_unroll_to_elements_1d(
// CHECK-SAME:    %[[ARG0:.+]]: vector<2xf32>
// CHECK:         %[[RES:.+]]:2 = vector.to_elements %[[ARG0]] : vector<2xf32>
// CHECK:         return %[[RES]]#0, %[[RES]]#1
func.func @negative_unroll_to_elements_1d(%arg0: vector<2xf32>) -> (f32, f32) {
  %0:2 = vector.to_elements %arg0 : vector<2xf32>
  return %0#0, %0#1 : f32, f32
}

// -----

// CHECK-LABEL: func.func @unroll_to_elements_2d(
// CHECK-SAME:    %[[ARG0:.+]]: vector<2x2xf32>
// CHECK:         %[[VEC0:.+]] = vector.extract %[[ARG0]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK:         %[[VEC1:.+]] = vector.extract %[[ARG0]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK:         %[[RES0:.+]]:2 = vector.to_elements %[[VEC0]] : vector<2xf32>
// CHECK:         %[[RES1:.+]]:2 = vector.to_elements %[[VEC1]] : vector<2xf32>
// CHECK:         return %[[RES0]]#0, %[[RES0]]#1, %[[RES1]]#0, %[[RES1]]#1
func.func @unroll_to_elements_2d(%arg0: vector<2x2xf32>) -> (f32, f32, f32, f32) {
  %0:4 = vector.to_elements %arg0 : vector<2x2xf32>
  return %0#0, %0#1, %0#2, %0#3 : f32, f32, f32, f32
}

// -----

// COM: Here we are testing the pattern ToElementsToTargetShape
// COM: The pattern has a native shape of [4], which means
// COM: that vectors multiples of 4 will be split. In this
// COM: case, that will happen in the function's body, not the argument.

// CHECK-LABEL: func.func @unroll_vector_8xf32
// CHECK-SAME: (%[[ARG0:.+]]: vector<8xf32>)
func.func @unroll_vector_8xf32(%arg0: vector<8xf32>) -> (f32, f32) {
  %0:8 = vector.to_elements %arg0 : vector<8xf32>

  // COM: We only return two elements, one from each of the
  // COM: vectors.
  return %0#3, %0#4: f32, f32

  // CHECK: %[[V0:.+]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
  // CHECK-NEXT: %[[V1:.+]] = vector.extract_strided_slice %[[ARG0]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
  // CHECK-NEXT: %[[ELEMS_0:.+]]:4 = vector.to_elements %[[V0]]
  // CHECK-NEXT: %[[ELEMS_1:.+]]:4 = vector.to_elements %[[V1]]
  // CHECK-NEXT: return %[[ELEMS_0]]#3, %[[ELEMS_1]]#0
}
