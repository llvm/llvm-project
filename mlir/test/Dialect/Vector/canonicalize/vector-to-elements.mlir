// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file -allow-unregistered-dialect | FileCheck %s

// This file contains some tests of folding/canonicalizing vector.to_elements

///===----------------------------------------------===//
///  Tests of `ToElementsOp::fold`
///===----------------------------------------------===//

// CHECK-LABEL: func @to_elements_of_shape_cast_folds
// CHECK-SAME: (%[[VEC:.*]]: vector<4xf32>) -> (f32, f32, f32, f32)
func.func @to_elements_of_shape_cast_folds(%v: vector<4xf32>) -> (f32, f32, f32, f32) {
  %sc = vector.shape_cast %v : vector<4xf32> to vector<2x2xf32>
  %e:4 = vector.to_elements %sc : vector<2x2xf32>
  // CHECK-NOT: vector.shape_cast
  // CHECK: %[[E:.*]]:4 = vector.to_elements %[[VEC]] : vector<4xf32>
  // CHECK: return %[[E]]#0, %[[E]]#1, %[[E]]#2, %[[E]]#3 : f32, f32, f32, f32
  return %e#0, %e#1, %e#2, %e#3 : f32, f32, f32, f32
}
