; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=VI -check-prefix=GCN %s

; Make sure there isn't an extra space between the instruction name and first operands.

; GCN-LABEL: {{^}}add_f32:
; SI: s_load_dword [[SREGA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x1c
; SI: s_load_dword [[SREGB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x13
; SI: v_mov_b32_e32 [[VREGA:v[0-9]+]], [[SREGA]]
; SI: v_add_f32_e32 [[RESULT:v[0-9]+]], [[SREGB]], [[VREGA]]

; VI-DAG: s_load_dword [[SREGA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x4c
; VI-DAG: s_load_dword [[SREGB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x70
; VI: v_mov_b32_e32 [[VREGB:v[0-9]+]], [[SREGB]]
; VI: v_add_f32_e32 [[RESULT:v[0-9]+]], [[SREGA]], [[VREGB]]

; GCN: buffer_store_dword [[RESULT]],
define amdgpu_kernel void @add_f32(ptr addrspace(1) %out, [8 x i32], float %a, [8 x i32], float %b) {
  %result = fadd float %a, %b
  store float %result, ptr addrspace(1) %out
  ret void
}
