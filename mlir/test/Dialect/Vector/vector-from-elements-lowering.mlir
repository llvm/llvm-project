// RUN: mlir-opt %s -test-unroll-vector-from-elements | FileCheck %s --check-prefix=CHECK-UNROLL

//===----------------------------------------------------------------------===//
// Test UnrollFromElements.
//===----------------------------------------------------------------------===//

// CHECK-UNROLL-LABEL: @unroll_from_elements_2d
// CHECK-UNROLL-SAME:    (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
// CHECK-UNROLL-NEXT:    %[[UNDEF_RES:.*]] = ub.poison : vector<2x2xf32>
// CHECK-UNROLL-NEXT:    %[[VEC_0:.*]] = vector.from_elements %[[ARG0]], %[[ARG1]] : vector<2xf32>
// CHECK-UNROLL-NEXT:    %[[RES_0:.*]] = vector.insert %[[VEC_0]], %[[UNDEF_RES]] [0] : vector<2xf32> into vector<2x2xf32>
// CHECK-UNROLL-NEXT:    %[[VEC_1:.*]] = vector.from_elements %[[ARG2]], %[[ARG3]] : vector<2xf32>
// CHECK-UNROLL-NEXT:    %[[RES_1:.*]] = vector.insert %[[VEC_1]], %[[RES_0]] [1] : vector<2xf32> into vector<2x2xf32>
// CHECK-UNROLL-NEXT:    return %[[RES_1]] : vector<2x2xf32>
func.func @unroll_from_elements_2d(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> vector<2x2xf32> {
  %0 = vector.from_elements %arg0, %arg1, %arg2, %arg3 : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// CHECK-UNROLL-LABEL: @unroll_from_elements_3d
// CHECK-UNROLL-SAME:    (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
// CHECK-UNROLL-NEXT:    %[[UNDEF_RES:.*]] = ub.poison : vector<2x1x2xf32>
// CHECK-UNROLL-NEXT:    %[[UNDEF_RANK_2:.*]] = ub.poison : vector<1x2xf32>
// CHECK-UNROLL-NEXT:    %[[VEC_0:.*]] = vector.from_elements %[[ARG0]], %[[ARG1]] : vector<2xf32>
// CHECK-UNROLL-NEXT:    %[[RANK_2_0:.*]] = vector.insert %[[VEC_0]], %[[UNDEF_RANK_2]] [0] : vector<2xf32> into vector<1x2xf32>
// CHECK-UNROLL-NEXT:    %[[RES_0:.*]] = vector.insert %[[RANK_2_0]], %[[UNDEF_RES]] [0] : vector<1x2xf32> into vector<2x1x2xf32>
// CHECK-UNROLL-NEXT:    %[[VEC_1:.*]] = vector.from_elements %[[ARG2]], %[[ARG3]] : vector<2xf32>
// CHECK-UNROLL-NEXT:    %[[RANK_2_1:.*]] = vector.insert %[[VEC_1]], %[[UNDEF_RANK_2]] [0] : vector<2xf32> into vector<1x2xf32>
// CHECK-UNROLL-NEXT:    %[[RES_1:.*]] = vector.insert %[[RANK_2_1]], %[[RES_0]] [1] : vector<1x2xf32> into vector<2x1x2xf32>
// CHECK-UNROLL-NEXT:    return %[[RES_1]] : vector<2x1x2xf32>
func.func @unroll_from_elements_3d(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> vector<2x1x2xf32> {
  %0 = vector.from_elements %arg0, %arg1, %arg2, %arg3 : vector<2x1x2xf32>
  return %0 : vector<2x1x2xf32>
}

// 1-D vector.from_elements should not be unrolled.

// CHECK-UNROLL-LABEL: @negative_unroll_from_elements_1d
// CHECK-UNROLL-SAME:    (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK-UNROLL-NEXT:         %[[RES:.*]] = vector.from_elements %[[ARG0]], %[[ARG1]] : vector<2xf32>
// CHECK-UNROLL-NEXT:    return %[[RES]] : vector<2xf32>
func.func @negative_unroll_from_elements_1d(%arg0: f32, %arg1: f32) -> vector<2xf32> {
  %0 = vector.from_elements %arg0, %arg1 : vector<2xf32>
  return %0 : vector<2xf32>
}
