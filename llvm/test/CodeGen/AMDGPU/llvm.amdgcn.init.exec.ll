;RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck %s --check-prefix=GCN

; GCN-LABEL: {{^}}full_mask:
; GCN: s_mov_b64 exec, -1
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @full_mask(float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec(i64 -1)
  ret float %s
}

; GCN-LABEL: {{^}}partial_mask:
; GCN: s_mov_b64 exec, 0x1e240
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @partial_mask(float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec(i64 123456)
  ret float %s
}

; GCN-LABEL: {{^}}input_s3off8:
; GCN: s_bfe_u32 s0, s3, 0x70008
; GCN: s_bfm_b64 exec, s0, 0
; GCN: s_cmp_eq_u32 s0, 64
; GCN: s_cmov_b64 exec, -1
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @input_s3off8(i32 inreg, i32 inreg, i32 inreg, i32 inreg %count, float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 8)
  ret float %s
}

; GCN-LABEL: {{^}}input_s0off19:
; GCN: s_bfe_u32 s0, s0, 0x70013
; GCN: s_bfm_b64 exec, s0, 0
; GCN: s_cmp_eq_u32 s0, 64
; GCN: s_cmov_b64 exec, -1
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @input_s0off19(i32 inreg %count, float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 19)
  ret float %s
}

; GCN-LABEL: {{^}}reuse_input:
; GCN: s_bfe_u32 s1, s0, 0x70013
; GCN: s_bfm_b64 exec, s1, 0
; GCN: s_cmp_eq_u32 s1, 64
; GCN: s_cmov_b64 exec, -1
; GCN: v_add_u32_e32 v0, s0, v0
define amdgpu_ps float @reuse_input(i32 inreg %count, i32 %a) {
main_body:
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 19)
  %s = add i32 %a, %count
  %f = sitofp i32 %s to float
  ret float %f
}

; GCN-LABEL: {{^}}reuse_input2:
; GCN: s_bfe_u32 s1, s0, 0x70013
; GCN: s_bfm_b64 exec, s1, 0
; GCN: s_cmp_eq_u32 s1, 64
; GCN: s_cmov_b64 exec, -1
; GCN: v_add_u32_e32 v0, s0, v0
define amdgpu_ps float @reuse_input2(i32 inreg %count, i32 %a) {
main_body:
  %s = add i32 %a, %count
  %f = sitofp i32 %s to float
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 19)
  ret float %f
}

; GCN-LABEL: {{^}}init_unreachable:
;
; This used to crash.
define amdgpu_ps void @init_unreachable() {
main_body:
  call void @llvm.amdgcn.init.exec(i64 -1)
  unreachable
}

declare void @llvm.amdgcn.init.exec(i64) #1
declare void @llvm.amdgcn.init.exec.from.input(i32, i32) #1

attributes #1 = { convergent }
