// RUN: mlir-opt %s -transform-preload-library='transform-library-paths=%p/td/inner-outer-dim-conversion.mlir' \
// RUN: -transform-interpreter=entry-point=inner_outer_dim_reduction_conversion | FileCheck %s

//===----------------------------------------------------------------------===//
// Test InnerOuterDimReductionConversion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @inner_outer_dim_reduction_conversion(
// CHECK-SAME: %[[ARG0:.+]]: vector<2x3x5x7xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<3x5xf32>
func.func @inner_outer_dim_reduction_conversion(%arg0: vector<2x3x5x7xf32>, %acc: vector<3x5xf32>) -> (vector<3x5xf32>) {
  // CHECK: %[[TRANSPOSE:.+]] = vector.transpose %[[ARG0]], [0, 3, 1, 2] : vector<2x3x5x7xf32> to vector<2x7x3x5xf32>
  // CHECK: %[[RES:.+]] = vector.multi_reduction <add>, %[[TRANSPOSE]], %[[ACC]] [0, 1] : vector<2x7x3x5xf32> to vector<3x5xf32>
  %1 = vector.multi_reduction <add>, %arg0, %acc [0, 3] : vector<2x3x5x7xf32> to vector<3x5xf32>

  // CHECK: return %[[RES]]
  return %1 : vector<3x5xf32>
}

