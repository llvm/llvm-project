// RUN: mlir-opt -split-input-file -convert-arith-to-spirv=enable-fast-math -verify-diagnostics %s | FileCheck %s

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, #spv.resource_limits<>>
} {

// CHECK-LABEL: @cmpf_ordered
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func.func @cmpf_ordered(%arg0 : f32, %arg1 : f32) -> i1 {
  // CHECK: %[[T:.+]] = spv.Constant true
  %0 = arith.cmpf ord, %arg0, %arg1 : f32
  // CHECK: return %[[T]]
  return %0: i1
}

// CHECK-LABEL: @cmpf_unordered
// CHECK-SAME: %[[LHS:.+]]: vector<4xf32>, %[[RHS:.+]]: vector<4xf32>
func.func @cmpf_unordered(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> vector<4xi1> {
  // CHECK: %[[F:.+]] = spv.Constant dense<false>
  %0 = arith.cmpf uno, %arg0, %arg1 : vector<4xf32>
  // CHECK: return %[[F]]
  return %0: vector<4xi1>
}

} // end module

// -----

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], []>, #spv.resource_limits<>>
} {

// CHECK-LABEL: @minf
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func.func @minf(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: %[[F:.+]] = spv.GL.FMin %[[LHS]], %[[RHS]]
  %0 = arith.minf %arg0, %arg1 : f32
  // CHECK: return %[[F]]
  return %0: f32
}

// CHECK-LABEL: @maxf
// CHECK-SAME: %[[LHS:.+]]: vector<4xf32>, %[[RHS:.+]]: vector<4xf32>
func.func @maxf(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[F:.+]] = spv.GL.FMax %[[LHS]], %[[RHS]]
  %0 = arith.maxf %arg0, %arg1 : vector<4xf32>
  // CHECK: return %[[F]]
  return %0: vector<4xf32>
}

} // end module
