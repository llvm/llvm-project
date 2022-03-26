; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -verify-machineinstrs < %s 2>%t.err | FileCheck %s
; RUN: FileCheck -check-prefix=ERR %s < %t.err

; This testcase fails register allocation at the same time it performs
; virtual register splitting (by introducing VGPR to AGPR copies). We
; still need to enqueue and allocate the newly split vregs after the
; failure.


; ERR: error: ran out of registers during register allocation
; ERR-NEXT: error: ran out of registers during register allocation
; ERR-NEXT: error: ran out of registers during register allocation
; ERR-NOT: ERROR

; CHECK: v_accvgpr_write_b32
; CHECK: v_accvgpr_write_b32
; CHECK: v_accvgpr_write_b32
; CHECK: v_accvgpr_write_b32
; CHECK: v_accvgpr_write_b32
; CHECK: v_accvgpr_write_b32
; CHECK: v_accvgpr_write_b32

; CHECK: v_accvgpr_read_b32
; CHECK: v_accvgpr_read_b32
; CHECK: v_accvgpr_read_b32
; CHECK: v_accvgpr_read_b32
; CHECK: v_accvgpr_read_b32
; CHECK: v_accvgpr_read_b32
; CHECK: v_accvgpr_read_b32
define amdgpu_kernel void @alloc_failure_with_split_vregs(float %v0, float %v1) #0 {
  %agpr0 = call float asm sideeffect "; def $0", "=${a0}"()
  %agpr.vec = insertelement <16 x float> undef, float %agpr0, i32 0
  %mfma0 = call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float %v0, float %v1, <16 x float> %agpr.vec, i32 0, i32 0, i32 0)
  %mfma0.3 = extractelement <16 x float> %mfma0, i32 3
  %insert = insertelement <16 x float> %mfma0, float %agpr0, i32 8
  %mfma1 = call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float %v0, float %v1, <16 x float> %insert, i32 0, i32 0, i32 0)
  %mfma1.3 = extractelement <16 x float> %mfma1, i32 3
  call void asm sideeffect "; use $0", "{a1}"(float %mfma1.3)
  ret void
}

declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float, float, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #1
declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { "amdgpu-waves-per-eu"="10,10" }
attributes #1 = { convergent nounwind readnone willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }
