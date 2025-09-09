// RUN: mlir-opt %s --test-vector-shuffle-lowering --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @shuffle_v1_smaller_arbitrary
// CHECK-SAME:    %[[V1:.*]]: vector<2xf32>, %[[V2:.*]]: vector<4xf32>
func.func @shuffle_v1_smaller_arbitrary(%v1: vector<2xf32>, %v2: vector<4xf32>) -> vector<5xf32> {
  // CHECK: %[[PROMOTE_V1:.*]] = vector.shuffle %[[V1]], %[[V1]] [0, 1, -1, -1] : vector<2xf32>, vector<2xf32>
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[PROMOTE_V1]], %[[V2]] [1, 5, 0, 6, 7] : vector<4xf32>, vector<4xf32>
  // CHECK: return %[[RESULT]] : vector<5xf32>
  %0 = vector.shuffle %v1, %v2 [1, 3, 0, 4, 5] : vector<2xf32>, vector<4xf32>
  return %0 : vector<5xf32>
}

// -----

// CHECK-LABEL: func.func @shuffle_v2_smaller_arbitrary
// CHECK-SAME:    %[[V1:.*]]: vector<4xi32>, %[[V2:.*]]: vector<2xi32>
func.func @shuffle_v2_smaller_arbitrary(%v1: vector<4xi32>, %v2: vector<2xi32>) -> vector<6xi32> {
  // CHECK: %[[PROMOTE_V2:.*]] = vector.shuffle %[[V2]], %[[V2]] [0, 1, -1, -1] : vector<2xi32>, vector<2xi32>
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[V1]], %[[PROMOTE_V2]] [3, 5, 1, 4, 0, 2] : vector<4xi32>, vector<4xi32>
  // CHECK: return %[[RESULT]] : vector<6xi32>
  %0 = vector.shuffle %v1, %v2 [3, 5, 1, 4, 0, 2] : vector<4xi32>, vector<2xi32>
  return %0 : vector<6xi32>
}

// -----

// CHECK-LABEL: func.func @shuffle_v1_smaller_concat
// CHECK-SAME:    %[[V1:.*]]: vector<3xf64>, %[[V2:.*]]: vector<5xf64>
func.func @shuffle_v1_smaller_concat(%v1: vector<3xf64>, %v2: vector<5xf64>) -> vector<8xf64> {
  // CHECK: %[[PROMOTE_V1:.*]] = vector.shuffle %[[V1]], %[[V1]] [0, 1, 2, -1, -1] : vector<3xf64>, vector<3xf64>
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[PROMOTE_V1]], %[[V2]] [0, 1, 2, 5, 6, 7, 8, 9] : vector<5xf64>, vector<5xf64>
  // CHECK: return %[[RESULT]] : vector<8xf64>
  %0 = vector.shuffle %v1, %v2 [0, 1, 2, 3, 4, 5, 6, 7] : vector<3xf64>, vector<5xf64>
  return %0 : vector<8xf64>
}

// -----

// CHECK-LABEL: func.func @shuffle_v2_smaller_concat
// CHECK-SAME:    %[[V1:.*]]: vector<4xi16>, %[[V2:.*]]: vector<2xi16>
func.func @shuffle_v2_smaller_concat(%v1: vector<4xi16>, %v2: vector<2xi16>) -> vector<6xi16> {
  // CHECK: %[[PROMOTE_V2:.*]] = vector.shuffle %[[V2]], %[[V2]] [0, 1, -1, -1] : vector<2xi16>, vector<2xi16>
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[V1]], %[[PROMOTE_V2]] [0, 1, 2, 3, 4, 5] : vector<4xi16>, vector<4xi16>
  // CHECK: return %[[RESULT]] : vector<6xi16>
  %0 = vector.shuffle %v1, %v2 [0, 1, 2, 3, 4, 5] : vector<4xi16>, vector<2xi16>
  return %0 : vector<6xi16>
}

// -----

// Test that shuffles with same size inputs are not modified.

// CHECK-LABEL: func.func @shuffle_same_input_sizes
// CHECK-SAME:    %[[V1:.*]]: vector<4xf32>, %[[V2:.*]]: vector<4xf32>
func.func @shuffle_same_input_sizes(%v1: vector<4xf32>, %v2: vector<4xf32>) -> vector<6xf32> {
  // CHECK-NOT: vector.shuffle %[[V1]], %[[V1]]
  // CHECK-NOT: vector.shuffle %[[V2]], %[[V2]]
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[V1]], %[[V2]] [0, 1, 4, 5, 2, 6] : vector<4xf32>, vector<4xf32>
  // CHECK: return %[[RESULT]] : vector<6xf32>
  %0 = vector.shuffle %v1, %v2 [0, 1, 4, 5, 2, 6] : vector<4xf32>, vector<4xf32>
  return %0 : vector<6xf32>
}

// -----

// Test that multi-dimensional shuffles are not modified.

// CHECK-LABEL: func.func @shuffle_2d_vectors_no_change
// CHECK-SAME:    %[[V1:.*]]: vector<2x4xf32>, %[[V2:.*]]: vector<3x4xf32>
func.func @shuffle_2d_vectors_no_change(%v1: vector<2x4xf32>, %v2: vector<3x4xf32>) -> vector<4x4xf32> {
  // CHECK-NOT: vector.shuffle %[[V1]], %[[V1]]
  // CHECK-NOT: vector.shuffle %[[V2]], %[[V2]]
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[V1]], %[[V2]] [0, 1, 2, 3] : vector<2x4xf32>, vector<3x4xf32>
  // CHECK: return %[[RESULT]] : vector<4x4xf32>
  %0 = vector.shuffle %v1, %v2 [0, 1, 2, 3] : vector<2x4xf32>, vector<3x4xf32>
  return %0 : vector<4x4xf32>
}
