; Enabling SPV_INTEL_rounded_divide_sqrt shouldn't add RoundedDivideSqrtINTEL,
; as FPRoundingMode on conversion instructions is supported in core SPIR-V

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - | FileCheck %s --implicit-check-not=RoundedDivideSqrtINTEL --implicit-check-not=SPV_INTEL_rounded_divide_sqrt
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - -filetype=obj | spirv-val %}

; The conversions still carry their rounding-mode decorations (core SPIR-V),
; both from the SPIR-V friendly and the OpenCL builtin form.
; CHECK-DAG: OpDecorate %[[#SPV_RTZ:]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#OCL_RTE:]] FPRoundingMode RTE
; CHECK: %[[#SPV_RTZ]] = OpConvertUToF
; CHECK: %[[#OCL_RTE]] = OpConvertUToF

define spir_kernel void @test_spirv_friendly(<2 x i32> %v, ptr addrspace(1) %p) {
entry:
  %cvt = call spir_func <2 x float> @_Z31__spirv_ConvertUToF_Rfloat2_rtzDv2_j(<2 x i32> %v)
  store <2 x float> %cvt, ptr addrspace(1) %p
  ret void
}

define spir_kernel void @test_ocl(<2 x i32> %v, ptr addrspace(1) %p) {
entry:
  %cvt = call spir_func <2 x float> @_Z18convert_float2_rteDv2_j(<2 x i32> %v)
  store <2 x float> %cvt, ptr addrspace(1) %p
  ret void
}

declare spir_func <2 x float> @_Z31__spirv_ConvertUToF_Rfloat2_rtzDv2_j(<2 x i32>)
declare spir_func <2 x float> @_Z18convert_float2_rteDv2_j(<2 x i32>)
