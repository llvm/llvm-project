; Enabling SPV_INTEL_rounded_divide_sqrt shouldn't add RoundedDivideSqrtINTEL,
; as FPRoundingMode on conversion instructions is supported in core SPIR-V

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - | FileCheck %s --implicit-check-not=RoundedDivideSqrtINTEL --implicit-check-not=SPV_INTEL_rounded_divide_sqrt
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - -filetype=obj | spirv-val %}

; The conversion still carries its rounding-mode decoration (core SPIR-V).
; CHECK: OpDecorate %[[#CVT:]] FPRoundingMode RTZ
; CHECK: %[[#CVT]] = OpConvertUToF

define spir_kernel void @test(<2 x i32> %v, ptr addrspace(1) %p) {
entry:
  %cvt = call spir_func <2 x float> @_Z31__spirv_ConvertUToF_Rfloat2_rtzDv2_j(<2 x i32> %v)
  store <2 x float> %cvt, ptr addrspace(1) %p
  ret void
}

declare spir_func <2 x float> @_Z31__spirv_ConvertUToF_Rfloat2_rtzDv2_j(<2 x i32>)
