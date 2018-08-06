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
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
entry:
  %a.val = load volatile half, half addrspace(1)* %a
  %b.val = load volatile half, half addrspace(1)* %b
  %c.val = load volatile half, half addrspace(1)* %c
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half %a.val, half %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_a
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; VI:  v_mov_b32_e32 v[[A_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_a(
    half addrspace(1)* %r,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
entry:
  %b.val = load volatile half, half addrspace(1)* %b
  %c.val = load volatile half, half addrspace(1)* %c
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half 3.0, half %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_b
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; VI:  v_mov_b32_e32 v[[B_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %c) {
entry:
  %a.val = load volatile half, half addrspace(1)* %a
  %c.val = load volatile half, half addrspace(1)* %c
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half %a.val, half 3.0, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_c
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; VI:  v_mov_b32_e32 v[[C_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_c(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load volatile half, half addrspace(1)* %a
  %b.val = load volatile half, half addrspace(1)* %b
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half %a.val, half %b.val, half 3.0)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_a_imm_b
; VI-DAG:  v_mov_b32_e32 v[[AB_F16:[0-9]+]], 0x4200{{$}}
; GCN-DAG: buffer_load_ushort v[[C_F16:[0-9]+]]
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[AB_F16]], v[[AB_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_a_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %c) {
entry:
  %c.val = load volatile half, half addrspace(1)* %c
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half 3.0, half 3.0, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_b_imm_c
; VI-DAG:  v_mov_b32_e32 v[[BC_F16:[0-9]+]], 0x4200{{$}}
; GCN-DAG: buffer_load_ushort v[[A_F16:[0-9]+]]
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[BC_F16]], v[[BC_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_b_imm_c(
    half addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half %a.val, half 3.0, half 3.0)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}div_fixup_f16_imm_a_imm_c
; VI-DAG:  v_mov_b32_e32 v[[AC_F16:[0-9]+]], 0x4200{{$}}
; GCN-DAG: buffer_load_ushort v[[B_F16:[0-9]+]]
; VI:  v_div_fixup_f16 v[[R_F16:[0-9]+]], v[[AC_F16]], v[[B_F16]], v[[AC_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @div_fixup_f16_imm_a_imm_c(
    half addrspace(1)* %r,
    half addrspace(1)* %b) {
entry:
  %b.val = load half, half addrspace(1)* %b
  %r.val = call half @llvm.amdgcn.div.fixup.f16(half 3.0, half %b.val, half 3.0)
  store half %r.val, half addrspace(1)* %r
  ret void
}
