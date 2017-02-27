; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; Make sure we don't crash or assert on spir_kernel calling convention.

; GCN-LABEL: {{^}}kernel:
; GCN: s_endpgm
define spir_kernel void @kernel(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; FIXME: This is treated like a kernel
; GCN-LABEL: {{^}}func:
; GCN: s_endpgm
define spir_func void @func(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ps_ret_cc_f16:
; SI: v_cvt_f16_f32_e32 v0, v0
; SI: v_cvt_f32_f16_e32 v0, v0
; SI: v_add_f32_e32 v0, 1.0, v0

; VI: v_add_f16_e32 v0, 1.0, v0
; VI: ; return
define amdgpu_ps half @ps_ret_cc_f16(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; GCN-LABEL: {{^}}ps_ret_cc_inreg_f16:
; SI: v_cvt_f16_f32_e32 v0, s0
; SI: v_cvt_f32_f16_e32 v0, v0
; SI: v_add_f32_e32 v0, 1.0, v0

; VI: v_add_f16_e64 v0, s0, 1.0
; VI: ; return
define amdgpu_ps half @ps_ret_cc_inreg_f16(half inreg %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}
