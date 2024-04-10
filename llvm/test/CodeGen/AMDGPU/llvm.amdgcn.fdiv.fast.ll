; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck %s

declare float @llvm.amdgcn.fdiv.fast(float, float) #0

; CHECK-LABEL: {{^}}test_fdiv_fast:
; CHECK: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, vcc
; CHECK: v_mul_f32_e32
; CHECK: v_rcp_f32_e32
; CHECK: v_mul_f32_e32
; CHECK: v_mul_f32_e32
define amdgpu_kernel void @test_fdiv_fast(ptr addrspace(1) %out, float %a, float %b) #1 {
  %fdiv = call float @llvm.amdgcn.fdiv.fast(float %a, float %b)
  store float %fdiv, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
