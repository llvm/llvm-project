; NOTE: Checks are intentionally hand-written to verify only instruction selection.
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s
; RUN: llc -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.00-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck --check-prefix=ERR %s
; RUN: not llc -global-isel=1 -global-isel-abort=0 -mtriple=amdgpu9.00-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck --check-prefix=ERR %s

; ERR: error: <unknown>:0:0: in function @fma_mix_f32 float (i32, i32, i32): llvm.amdgcn.fma.mix.f32 requires target feature 'fma-mix-insts'
; ERR: error: <unknown>:0:0: in function @fma_mix_f32_bf16 float (i32, i32, i32): llvm.amdgcn.fma.mix.f32.bf16 requires target feature 'fma-mix-bf16-insts'

declare float @llvm.amdgcn.fma.mix.f32(i32, i32, i32, i32 immarg, i32 immarg, i32 immarg)
declare float @llvm.amdgcn.fma.mix.f32.bf16(i32, i32, i32, i32 immarg, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.fma.mixlo.f16(i32, i32, i32, i32, i32 immarg, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.fma.mixhi.f16(i32, i32, i32, i32, i32 immarg, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.fma.mixlo.bf16(i32, i32, i32, i32, i32 immarg, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.fma.mixhi.bf16(i32, i32, i32, i32, i32 immarg, i32 immarg, i32 immarg)

define float @fma_mix_f32(i32 %src0, i32 %src1, i32 %src2) {
; CHECK-LABEL: fma_mix_f32:
; CHECK:       v_fma_mix_f32 v0, v0, v1, v2 op_sel:[0,1,0] op_sel_hi:[0,0,1]
  %result = call float @llvm.amdgcn.fma.mix.f32(i32 %src0, i32 %src1, i32 %src2, i32 0, i32 1, i32 2)
  ret float %result
}

define float @fma_mix_f32_bf16(i32 %src0, i32 %src1, i32 %src2) {
; CHECK-LABEL: fma_mix_f32_bf16:
; CHECK:       v_fma_mix_f32_bf16 v0, v0, v1, v2 op_sel:[1,0,1] op_sel_hi:[1,1,0]
  %result = call float @llvm.amdgcn.fma.mix.f32.bf16(i32 %src0, i32 %src1, i32 %src2, i32 3, i32 2, i32 1)
  ret float %result
}

define i32 @fma_mixlo_f16(i32 %src0, i32 %src1, i32 %src2, i32 %dst) {
; CHECK-LABEL: fma_mixlo_f16:
; CHECK:       v_fma_mixlo_f16 v3, v0, v1, v2 op_sel_hi:[1,1,1]
  %result = call i32 @llvm.amdgcn.fma.mixlo.f16(i32 %src0, i32 %src1, i32 %src2, i32 %dst, i32 2, i32 2, i32 2)
  ret i32 %result
}

define i32 @fma_mixhi_f16(i32 %src0, i32 %src1, i32 %src2, i32 %dst) {
; CHECK-LABEL: fma_mixhi_f16:
; CHECK:       v_fma_mixhi_f16 v3, v0, v1, v2 op_sel_hi:[1,1,1]
  %result = call i32 @llvm.amdgcn.fma.mixhi.f16(i32 %src0, i32 %src1, i32 %src2, i32 %dst, i32 2, i32 2, i32 2)
  ret i32 %result
}

define i32 @fma_mixlo_bf16(i32 %src0, i32 %src1, i32 %src2, i32 %dst) {
; CHECK-LABEL: fma_mixlo_bf16:
; CHECK:       v_fma_mixlo_bf16 v3, v0, v1, v2 op_sel_hi:[1,1,1]
  %result = call i32 @llvm.amdgcn.fma.mixlo.bf16(i32 %src0, i32 %src1, i32 %src2, i32 %dst, i32 2, i32 2, i32 2)
  ret i32 %result
}

define i32 @fma_mixhi_bf16(i32 %src0, i32 %src1, i32 %src2, i32 %dst) {
; CHECK-LABEL: fma_mixhi_bf16:
; CHECK:       v_fma_mixhi_bf16 v3, v0, v1, v2 op_sel_hi:[1,1,1]
  %result = call i32 @llvm.amdgcn.fma.mixhi.bf16(i32 %src0, i32 %src1, i32 %src2, i32 %dst, i32 2, i32 2, i32 2)
  ret i32 %result
}
