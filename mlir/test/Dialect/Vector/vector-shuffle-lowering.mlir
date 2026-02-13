// RUN: mlir-opt %s --test-vector-shuffle-lowering --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @shuffle_smaller_lhs_arbitrary
// CHECK-SAME:    %[[LHS:.*]]: vector<2xf32>, %[[RHS:.*]]: vector<4xf32>
func.func @shuffle_smaller_lhs_arbitrary(%lhs: vector<2xf32>, %rhs: vector<4xf32>) -> vector<5xf32> {
  // CHECK: %[[PROMOTE_LHS:.*]] = vector.shuffle %[[LHS]], %[[LHS]] [0, 1, -1, -1] : vector<2xf32>, vector<2xf32>
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[PROMOTE_LHS]], %[[RHS]] [1, 5, 0, 6, 7] : vector<4xf32>, vector<4xf32>
  // CHECK: return %[[RESULT]] : vector<5xf32>
  %0 = vector.shuffle %lhs, %rhs [1, 3, 0, 4, 5] : vector<2xf32>, vector<4xf32>
  return %0 : vector<5xf32>
}

// -----

// CHECK-LABEL: func.func @shuffle_smaller_rhs_arbitrary
// CHECK-SAME:    %[[LHS:.*]]: vector<4xi32>, %[[RHS:.*]]: vector<2xi32>
func.func @shuffle_smaller_rhs_arbitrary(%lhs: vector<4xi32>, %rhs: vector<2xi32>) -> vector<6xi32> {
  // CHECK: %[[PROMOTE_RHS:.*]] = vector.shuffle %[[RHS]], %[[RHS]] [0, 1, -1, -1] : vector<2xi32>, vector<2xi32>
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[LHS]], %[[PROMOTE_RHS]] [3, 5, 1, 4, 0, 2] : vector<4xi32>, vector<4xi32>
  // CHECK: return %[[RESULT]] : vector<6xi32>
  %0 = vector.shuffle %lhs, %rhs [3, 5, 1, 4, 0, 2] : vector<4xi32>, vector<2xi32>
  return %0 : vector<6xi32>
}

// -----

// CHECK-LABEL: func.func @shuffle_smaller_lhs_concat
// CHECK-SAME:    %[[LHS:.*]]: vector<3xf64>, %[[RHS:.*]]: vector<5xf64>
func.func @shuffle_smaller_lhs_concat(%lhs: vector<3xf64>, %rhs: vector<5xf64>) -> vector<8xf64> {
  // CHECK: %[[PROMOTE_LHS:.*]] = vector.shuffle %[[LHS]], %[[LHS]] [0, 1, 2, -1, -1] : vector<3xf64>, vector<3xf64>
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[PROMOTE_LHS]], %[[RHS]] [0, 1, 2, 5, 6, 7, 8, 9] : vector<5xf64>, vector<5xf64>
  // CHECK: return %[[RESULT]] : vector<8xf64>
  %0 = vector.shuffle %lhs, %rhs [0, 1, 2, 3, 4, 5, 6, 7] : vector<3xf64>, vector<5xf64>
  return %0 : vector<8xf64>
}

// -----

// CHECK-LABEL: func.func @shuffle_smaller_rhs_concat
// CHECK-SAME:    %[[LHS:.*]]: vector<4xi16>, %[[RHS:.*]]: vector<2xi16>
func.func @shuffle_smaller_rhs_concat(%lhs: vector<4xi16>, %rhs: vector<2xi16>) -> vector<6xi16> {
  // CHECK: %[[PROMOTE_RHS:.*]] = vector.shuffle %[[RHS]], %[[RHS]] [0, 1, -1, -1] : vector<2xi16>, vector<2xi16>
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[LHS]], %[[PROMOTE_RHS]] [0, 1, 2, 3, 4, 5] : vector<4xi16>, vector<4xi16>
  // CHECK: return %[[RESULT]] : vector<6xi16>
  %0 = vector.shuffle %lhs, %rhs [0, 1, 2, 3, 4, 5] : vector<4xi16>, vector<2xi16>
  return %0 : vector<6xi16>
}

// -----

// Test that shuffles with same size inputs are not modified.

// CHECK-LABEL: func.func @negative_shuffle_same_input_sizes
// CHECK-SAME:    %[[LHS:.*]]: vector<4xf32>, %[[RHS:.*]]: vector<4xf32>
func.func @negative_shuffle_same_input_sizes(%lhs: vector<4xf32>, %rhs: vector<4xf32>) -> vector<6xf32> {
  // CHECK-NOT: vector.shuffle %[[LHS]], %[[LHS]]
  // CHECK-NOT: vector.shuffle %[[RHS]], %[[RHS]]
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[LHS]], %[[RHS]] [0, 1, 4, 5, 2, 6] : vector<4xf32>, vector<4xf32>
  // CHECK: return %[[RESULT]] : vector<6xf32>
  %0 = vector.shuffle %lhs, %rhs [0, 1, 4, 5, 2, 6] : vector<4xf32>, vector<4xf32>
  return %0 : vector<6xf32>
}

// -----

// Test that multi-dimensional shuffles are not modified.

// CHECK-LABEL: func.func @negative_shuffle_2d_vectors
// CHECK-SAME:    %[[LHS:.*]]: vector<2x4xf32>, %[[RHS:.*]]: vector<3x4xf32>
func.func @negative_shuffle_2d_vectors(%lhs: vector<2x4xf32>, %rhs: vector<3x4xf32>) -> vector<4x4xf32> {
  // CHECK-NOT: vector.shuffle %[[LHS]], %[[LHS]]
  // CHECK-NOT: vector.shuffle %[[RHS]], %[[RHS]]
  // CHECK: %[[RESULT:.*]] = vector.shuffle %[[LHS]], %[[RHS]] [0, 1, 2, 3] : vector<2x4xf32>, vector<3x4xf32>
  // CHECK: return %[[RESULT]] : vector<4x4xf32>
  %0 = vector.shuffle %lhs, %rhs [0, 1, 2, 3] : vector<2x4xf32>, vector<3x4xf32>
  return %0 : vector<4x4xf32>
}
