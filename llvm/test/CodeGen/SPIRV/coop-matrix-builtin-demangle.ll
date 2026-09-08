; RUN: llc -mtriple=spirv64-unknown-unknown \
; RUN:   -spirv-ext=+SPV_KHR_cooperative_matrix \
; RUN:   -filetype=asm < %s | FileCheck %s

target triple = "spirv64-unknown-unknown"

%coopmat.f32.sc3.16x16.u0 =
    type target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)

; ---------------------------------------------------------------------------
; Suffix branch.
; ---------------------------------------------------------------------------

declare spir_func %coopmat.f32.sc3.16x16.u0
@__spirv_CooperativeMatrixLoadKHR_f32_sc3_16x16_u0_global(
    ptr addrspace(1), i32)

define spir_kernel void @test_mangled_load(
    ptr addrspace(1) %ptr) {
entry:
  %mat = call spir_func %coopmat.f32.sc3.16x16.u0
      @__spirv_CooperativeMatrixLoadKHR_f32_sc3_16x16_u0_global(
          ptr addrspace(1) %ptr, i32 16)
  ret void
}

; CHECK-LABEL: ; -- Begin function test_mangled_load
; CHECK:       OpCooperativeMatrixLoadKHR
; CHECK:       OpFunctionEnd

; ---------------------------------------------------------------------------
; No-suffix branch.
; ---------------------------------------------------------------------------

declare spir_func %coopmat.f32.sc3.16x16.u0
@__spirv_CooperativeMatrixLoadKHR(
    ptr addrspace(1), i32)

define spir_kernel void @test_canonical_load(
    ptr addrspace(1) %ptr) {
entry:
  %mat = call spir_func %coopmat.f32.sc3.16x16.u0
      @__spirv_CooperativeMatrixLoadKHR(
          ptr addrspace(1) %ptr, i32 16)
  ret void
}

; CHECK-LABEL: ; -- Begin function test_canonical_load
; CHECK:       OpCooperativeMatrixLoadKHR
; CHECK:       OpFunctionEnd

