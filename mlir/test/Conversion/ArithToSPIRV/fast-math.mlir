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

// -----

// FPFastMathMode decoration tests (requires Kernel capability)

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @addf_fast_math
func.func @addf_fast_math(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}} {fp_fast_math_mode = #spirv.fastmath_mode<NotNaN|NotInf>} : f32
  %0 = arith.addf %arg0, %arg1 fastmath<nnan,ninf> : f32
  return %0: f32
}

// CHECK-LABEL: @mulf_no_fast_math
func.func @mulf_no_fast_math(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: spirv.FMul %{{.*}}, %{{.*}} : f32
  // CHECK-NOT: fp_fast_math_mode
  %0 = arith.mulf %arg0, %arg1 : f32
  return %0: f32
}

// CHECK-LABEL: @subf_all_flags
func.func @subf_all_flags(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: spirv.FSub %{{.*}}, %{{.*}} {fp_fast_math_mode = #spirv.fastmath_mode<NotNaN|NotInf|NSZ|AllowRecip>} : f32
  %0 = arith.subf %arg0, %arg1 fastmath<fast> : f32
  return %0: f32
}

// CHECK-LABEL: @negf_fast_math
func.func @negf_fast_math(%arg0 : f32) -> f32 {
  // CHECK: spirv.FNegate %{{.*}} {fp_fast_math_mode = #spirv.fastmath_mode<NSZ>} : f32
  %0 = arith.negf %arg0 fastmath<nsz> : f32
  return %0: f32
}

// CHECK-LABEL: @divf_fast_math
func.func @divf_fast_math(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: spirv.FDiv %{{.*}}, %{{.*}} {fp_fast_math_mode = #spirv.fastmath_mode<AllowRecip>} : f32
  %0 = arith.divf %arg0, %arg1 fastmath<arcp> : f32
  return %0: f32
}

// CHECK-LABEL: @remf_fast_math
func.func @remf_fast_math(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: spirv.FRem %{{.*}}, %{{.*}} {fp_fast_math_mode = #spirv.fastmath_mode<NotNaN>} : f32
  %0 = arith.remf %arg0, %arg1 fastmath<nnan> : f32
  return %0: f32
}

// Test that unsupported flags (reassoc, contract, afn) are silently dropped
// CHECK-LABEL: @addf_unsupported_flags_only
func.func @addf_unsupported_flags_only(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}} : f32
  // CHECK-NOT: fp_fast_math_mode
  %0 = arith.addf %arg0, %arg1 fastmath<reassoc,contract,afn> : f32
  return %0: f32
}

// CHECK-LABEL: @addf_vector_fast_math
func.func @addf_vector_fast_math(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> vector<4xf32> {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}} {fp_fast_math_mode = #spirv.fastmath_mode<NotNaN>} : vector<4xf32>
  %0 = arith.addf %arg0, %arg1 fastmath<nnan> : vector<4xf32>
  return %0: vector<4xf32>
}

} // end module

// -----

// FPFastMathMode decoration requires the Kernel capability. Without it the decoration is dropped.

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @addf_fast_math_no_kernel
func.func @addf_fast_math_no_kernel(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}} : f32
  // CHECK-NOT: fp_fast_math_mode
  %0 = arith.addf %arg0, %arg1 fastmath<nnan,ninf> : f32
  return %0: f32
}

} // end module
