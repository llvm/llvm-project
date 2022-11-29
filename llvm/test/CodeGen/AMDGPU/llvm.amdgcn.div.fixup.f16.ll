; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare half @llvm.amdgcn.div.fixup.f16(half %a, half %b, half %c)

; GCN-LABEL: {{^}}div_fixup_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load volatile half, ptr addrspace(1) %a
  %b.val = load volatile half, ptr addrspace(1) %b
  %c.val = load volatile half, ptr addrspace(1) %c
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half %a.val, half %b.val, half %c.val)
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_a
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; VI:  s_movk_i32 s[[A_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], s[[A_F16]], v[[B_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_a(
    ptr addrspace(1) %r,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %b.val = load volatile half, ptr addrspace(1) %b
  %c.val = load volatile half, ptr addrspace(1) %c
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half 3.0, half %b.val, half %c.val)
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_b
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; VI:  s_movk_i32 s[[B_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[A_F16]], s[[B_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %c) {
entry:
  %a.val = load volatile half, ptr addrspace(1) %a
  %c.val = load volatile half, ptr addrspace(1) %c
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half %a.val, half 3.0, half %c.val)
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_c
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; VI:  s_movk_i32 s[[C_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], s[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_c(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) {
entry:
  %a.val = load volatile half, ptr addrspace(1) %a
  %b.val = load volatile half, ptr addrspace(1) %b
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half %a.val, half %b.val, half 3.0)
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_a_imm_b
; VI-DAG:  s_movk_i32 [[AB_F16:s[0-9]+]], 0x4200{{$}}
; GCN-DAG: buffer_load_ushort v[[C_F16:[0-9]+]]
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], [[AB_F16]], [[AB_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_a_imm_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %c) {
entry:
  %c.val = load volatile half, ptr addrspace(1) %c
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half 3.0, half 3.0, half %c.val)
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_b_imm_c
; VI-DAG:  s_movk_i32 [[BC_F16:s[0-9]+]], 0x4200{{$}}
; GCN-DAG: buffer_load_ushort v[[A_F16:[0-9]+]]
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[A_F16]], [[BC_F16]], [[BC_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_b_imm_c(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half %a.val, half 3.0, half 3.0)
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_a_imm_c
; VI-DAG:  s_movk_i32 [[AC_F16:s[0-9]+]], 0x4200{{$}}
; GCN-DAG: buffer_load_ushort v[[B_F16:[0-9]+]]
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], [[AC_F16]], v[[B_F16]], [[AC_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_a_imm_c(
    ptr addrspace(1) %r,
    ptr addrspace(1) %b) {
entry:
  %b.val = load half, ptr addrspace(1) %b
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half 3.0, half %b.val, half 3.0)
  store half %r.val, ptr addrspace(1) %r
  ret void
}
