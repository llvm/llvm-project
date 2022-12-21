; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s


; GCN-LABEL: {{^}}shader_cc:
; GCN: v_add_{{[iu]}}32_e32 v0, vcc, s8, v0
define amdgpu_cs float @shader_cc(<4 x i32> inreg, <4 x i32> inreg, i32 inreg %w, float %v) {
  %vi = bitcast float %v to i32
  %x = add i32 %vi, %w
  %xf = bitcast i32 %x to float
  ret float %xf
}

; GCN-LABEL: {{^}}kernel_cc:
; GCN: s_endpgm
define amdgpu_kernel void @kernel_cc(<4 x i32> inreg, <4 x i32> inreg, i32 inreg %w, float %v) {
  %vi = bitcast float %v to i32
  %x = add i32 %vi, %w
  %xf = bitcast i32 %x to float
  store float %xf, ptr addrspace(1) undef
  ret void
}
