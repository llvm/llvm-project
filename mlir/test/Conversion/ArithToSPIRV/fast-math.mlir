// RUN: mlir-opt -split-input-file -convert-arith-to-spirv -verify-diagnostics %s | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @cmpf_ordered
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func.func @cmpf_ordered(%arg0 : f32, %arg1 : f32) -> i1 {
  // CHECK: %[[T:.+]] = spirv.Constant true
  %0 = arith.cmpf ord, %arg0, %arg1 fastmath<fast> : f32
  // CHECK: return %[[T]]
  return %0: i1
}

// CHECK-LABEL: @cmpf_unordered
// CHECK-SAME: %[[LHS:.+]]: vector<4xf32>, %[[RHS:.+]]: vector<4xf32>
func.func @cmpf_unordered(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> vector<4xi1> {
  // CHECK: %[[F:.+]] = spirv.Constant dense<false>
  %0 = arith.cmpf uno, %arg0, %arg1 fastmath<nnan> : vector<4xf32>
  // CHECK: return %[[F]]
  return %0: vector<4xi1>
}

} // end module

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @minimumf
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func.func @minimumf(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: %[[F:.+]] = spirv.GL.FMin %[[LHS]], %[[RHS]]
  %0 = arith.minimumf %arg0, %arg1 fastmath<fast> : f32
  // CHECK: return %[[F]]
  return %0: f32
}

// CHECK-LABEL: @maximumf
// CHECK-SAME: %[[LHS:.+]]: vector<4xf32>, %[[RHS:.+]]: vector<4xf32>
func.func @maximumf(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[F:.+]] = spirv.GL.FMax %[[LHS]], %[[RHS]]
  %0 = arith.maximumf %arg0, %arg1 fastmath<fast> : vector<4xf32>
  // CHECK: return %[[F]]
  return %0: vector<4xf32>
}

// CHECK-LABEL: @minnumf
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func.func @minnumf(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: %[[F:.+]] = spirv.GL.FMin %[[LHS]], %[[RHS]]
  %0 = arith.minnumf %arg0, %arg1 fastmath<fast> : f32
  // CHECK: return %[[F]]
  return %0: f32
}

// CHECK-LABEL: @maxnumf
// CHECK-SAME: %[[LHS:.+]]: vector<4xf32>, %[[RHS:.+]]: vector<4xf32>
func.func @maxnumf(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[F:.+]] = spirv.GL.FMax %[[LHS]], %[[RHS]]
  %0 = arith.maxnumf %arg0, %arg1 fastmath<fast> : vector<4xf32>
  // CHECK: return %[[F]]
  return %0: vector<4xf32>
}

} // end module
