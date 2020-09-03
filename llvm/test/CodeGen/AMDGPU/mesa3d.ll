; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}scratch_ps:
; GCN: s_load_dwordx2 s[4:5], s[0:1], 0x0{{$}}
; GCN-DAG: s_mov_b32 s6, -1{{$}}
; GCN-DAG: s_mov_b32 s7, 0xe8f000
; GCN-DAG: v_mov_b32_e32 [[V:v[0-9]+]], 2
; GCN: buffer_store_dword [[V]], off, s[4:7], 0 offset:4
define amdgpu_ps void @scratch_ps(i32 addrspace(1)* %out, i32 %in) {
entry:
  %alloca = alloca i32, addrspace(5)
  store volatile i32 2, i32 addrspace(5)* %alloca
  ret void
}
