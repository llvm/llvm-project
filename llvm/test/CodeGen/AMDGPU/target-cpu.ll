; RUN: llc -mtriple=amdgpu9 < %s | FileCheck %s

; The default gfx9-generic subtarget lacks the mad-mix instructions, so the
; f16 multiply-add is expanded to conversions plus a separate fma.
; CHECK-LABEL: {{^}}target_none:
; CHECK: v_cvt_f32_f16_e32 [[A:v[0-9]+]], v0
; CHECK: v_cvt_f32_f16_e32 [[B:v[0-9]+]], v1
; CHECK: v_cvt_f32_f16_e32 [[C:v[0-9]+]], v2
; CHECK: v_mac_f32_e32 [[C]], [[A]], [[B]]
define float @target_none(half %a, half %b, half %c) #1 {
  %ae = fpext half %a to float
  %be = fpext half %b to float
  %ce = fpext half %c to float
  %r = call float @llvm.fmuladd.f32(float %ae, float %be, float %ce)
  ret float %r
}

; gfx900 has the mad-mix instructions, so the same multiply-add folds the f16
; conversions into a single v_mad_mix_f32.
; CHECK-LABEL: {{^}}target_gfx900:
; CHECK: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1]
define float @target_gfx900(half %a, half %b, half %c) #2 {
  %ae = fpext half %a to float
  %be = fpext half %b to float
  %ce = fpext half %c to float
  %r = call float @llvm.fmuladd.f32(float %ae, float %be, float %ce)
  ret float %r
}

; gfx906 has the fma-mix instructions, so the same multiply-add folds the f16
; conversions into a single v_fma_mix_f32.
; CHECK-LABEL: {{^}}target_gfx906_mad_mix:
; CHECK: v_fma_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1]
define float @target_gfx906_mad_mix(half %a, half %b, half %c) #4 {
  %ae = fpext half %a to float
  %be = fpext half %b to float
  %ce = fpext half %c to float
  %r = call float @llvm.fmuladd.f32(float %ae, float %be, float %ce)
  ret float %r
}

; The fdot2 intrinsic is only available on gfx906.
; CHECK-LABEL: {{^}}target_gfx906:
; CHECK: v_dot2_f32_f16
define amdgpu_kernel void @target_gfx906(ptr addrspace(1) %out, <2 x half> %a, <2 x half> %b, float %c) #3 {
  %dot = call float @llvm.amdgcn.fdot2(<2 x half> %a, <2 x half> %b, float %c, i1 false)
  store float %dot, ptr addrspace(1) %out, align 4
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float) #0
declare float @llvm.amdgcn.fdot2(<2 x half>, <2 x half>, float, i1 immarg) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind denormal_fpenv(float: preservesign) }
attributes #2 = { nounwind denormal_fpenv(float: preservesign) "target-cpu"="gfx900" }
attributes #3 = { nounwind "target-cpu"="gfx906" }
attributes #4 = { nounwind denormal_fpenv(float: preservesign) "target-cpu"="gfx906" }
