; RUN: opt -mtriple=amdgcn-- -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s

; CHECK: for function 'interp_p1_f16'
; CHECK: DIVERGENT:       %p1 = call float @llvm.amdgcn.interp.p1.f16
define amdgpu_ps float @interp_p1_f16(float inreg %i, float inreg %j, i32 inreg %m0) #0 {
main_body:
  %p1 = call float @llvm.amdgcn.interp.p1.f16(float %i, i32 1, i32 2, i1 0, i32 %m0)
  ret float %p1
}

; CHECK: for function 'interp_p2_f16'
; CHECK: DIVERGENT:       %p2 = call half @llvm.amdgcn.interp.p2.f16
define amdgpu_ps half @interp_p2_f16(float inreg %i, float inreg %j, i32 inreg %m0) #0 {
main_body:
  %p2 = call half @llvm.amdgcn.interp.p2.f16(float %i, float %j, i32 1, i32 2, i1 0, i32 %m0)
  ret half %p2
}

; float @llvm.amdgcn.interp.p1.f16(i, attrchan, attr, high, m0)
declare float @llvm.amdgcn.interp.p1.f16(float, i32, i32, i1, i32) #0
; half @llvm.amdgcn.interp.p1.f16(p1, j, attrchan, attr, high, m0)
declare half @llvm.amdgcn.interp.p2.f16(float, float, i32, i32, i1, i32) #0
declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #0

attributes #0 = { nounwind readnone }
