; RUN: llc < %s -mtriple=amdgpu6.00 | FileCheck %s -check-prefix=SI -check-prefix=FUNC
; RUN: llc < %s -mtriple=amdgpu8.02 | FileCheck %s -check-prefix=SI -check-prefix=FUNC

;FUNC-LABEL: test
;SI: v_cos_f32
;SI-NOT: v_cos_f32
define amdgpu_kernel void @test(ptr addrspace(1) %out, float %x) #1 {
   %cos = call float @llvm.cos.f32(float %x)
   store float %cos, ptr addrspace(1) %out
   ret void
}

;FUNC-LABEL: testv
;SI: v_cos_f32
;SI: v_cos_f32
;SI: v_cos_f32
;SI: v_cos_f32
;SI-NOT: v_cos_f32
define amdgpu_kernel void @testv(ptr addrspace(1) %out, <4 x float> inreg %vx) #1 {
   %cos = call <4 x float> @llvm.cos.v4f32(<4 x float> %vx)
   store <4 x float> %cos, ptr addrspace(1) %out
   ret void
}

declare float @llvm.cos.f32(float) readnone
declare <4 x float> @llvm.cos.v4f32(<4 x float>) readnone
