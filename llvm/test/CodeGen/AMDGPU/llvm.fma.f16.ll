; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare half @llvm.fma.f16(half %a, half %b, half %c)
declare <2 x half> @llvm.fma.v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)

; GCN-LABEL: {{^}}fma_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_fma_f32 v[[R_F32:[0-9]+]], v[[A_F32:[0-9]]], v[[B_F32:[0-9]]], v[[C_F32:[0-9]]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_fma_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fma_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %r.val = call half @llvm.fma.f16(half %a.val, half %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fma_f16_imm_a
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]

; SI:  v_mov_b32_e32 v[[A_F32:[0-9]+]], 0x40400000{{$}}
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_fma_f32 v[[R_F32:[0-9]+]], v[[A_F32:[0-9]]], v[[B_F32:[0-9]]], v[[C_F32:[0-9]]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_mov_b32_e32 v[[A_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_fma_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fma_f16_imm_a(
    half addrspace(1)* %r,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %r.val = call half @llvm.fma.f16(half 3.0, half %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fma_f16_imm_b
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; SI:  v_mov_b32_e32 v[[B_F32:[0-9]+]], 0x40400000{{$}}
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_fma_f32 v[[R_F32:[0-9]+]], v[[A_F32:[0-9]]], v[[B_F32:[0-9]]], v[[C_F32:[0-9]]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_mov_b32_e32 v[[B_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_fma_f16 v[[R_F16:[0-9]+]], v[[B_F16]], v[[A_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fma_f16_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %c) {
  %a.val = load half, half addrspace(1)* %a
  %c.val = load half, half addrspace(1)* %c
  %r.val = call half @llvm.fma.f16(half %a.val, half 3.0, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fma_f16_imm_c
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_mov_b32_e32 v[[C_F32:[0-9]+]], 0x40400000{{$}}
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_fma_f32 v[[R_F32:[0-9]+]], v[[A_F32:[0-9]]], v[[B_F32:[0-9]]], v[[C_F32:[0-9]]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_mov_b32_e32 v[[C_F16:[0-9]+]], 0x4200{{$}}
; VI:  v_fma_f16 v[[R_F16:[0-9]+]], v[[B_F16]], v[[A_F16]], v[[C_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fma_f16_imm_c(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = call half @llvm.fma.f16(half %a.val, half %b.val, half 3.0)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fma_v2f16
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[C_V2_F16:[0-9]+]]

; SI: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI: v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]

; SI: v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]

; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_F16_1]]
; SI-DAG:  v_fma_f32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], v[[B_F32_0]], v[[C_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_fma_f32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], v[[B_F32_1]], v[[C_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]

; VI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; VI: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; VI: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; VI: v_fma_f16 v[[R_F16_0:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]], v[[C_V2_F16]]
; VI: v_fma_f16 v[[R_F16_1:[0-9]+]], v[[A_F16_1]], v[[B_F16_1]], v[[C_F16_1]]

; GCN: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; GCN: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_HI]], v[[R_F16_0]]
; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fma_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) {
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %r.val = call <2 x half> @llvm.fma.v2f16(<2 x half> %a.val, <2 x half> %b.val, <2 x half> %c.val)
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fma_v2f16_imm_a:
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[C_V2_F16:[0-9]+]]
; SI:  v_mov_b32_e32 v[[A_F32:[0-9]+]], 0x40400000{{$}}
; VI:  v_mov_b32_e32 v[[A_F16:[0-9]+]], 0x4200{{$}}
; GCN-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; GCN-DAG: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_F16_1]]
; SI:  v_fma_f32 v[[R_F32_0:[0-9]+]], v[[B_F32_0]], v[[A_F32]], v[[C_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_fma_f32 v[[R_F32_1:[0-9]+]], v[[B_F32_1]], v[[A_F32]], v[[C_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]

; VI:  v_fma_f16 v[[R_F16_0:[0-9]+]], v[[B_V2_F16]], v[[A_F16]], v[[C_V2_F16]]
; VI:  v_fma_f16 v[[R_F16_1:[0-9]+]], v[[B_F16_1]], v[[A_F16]], v[[C_F16_1]]
; GCN: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; GCN: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_HI]], v[[R_F16_0]]
; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fma_v2f16_imm_a(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) {
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %r.val = call <2 x half> @llvm.fma.v2f16(<2 x half> <half 3.0, half 3.0>, <2 x half> %b.val, <2 x half> %c.val)
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fma_v2f16_imm_b:
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[C_V2_F16:[0-9]+]]
; SI:  v_mov_b32_e32 v[[B_F32:[0-9]+]], 0x40400000{{$}}
; VI:  v_mov_b32_e32 v[[B_F16:[0-9]+]], 0x4200{{$}}

; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_F16_1]]
; SI-DAG:  v_fma_f32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], v[[B_F32]], v[[C_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_fma_f32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], v[[B_F32]], v[[C_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]

; VI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; VI: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; VI:  v_fma_f16 v[[R_F16_0:[0-9]+]], v[[A_V2_F16]], v[[B_F16]], v[[C_V2_F16]]
; VI:  v_fma_f16 v[[R_F16_1:[0-9]+]], v[[A_F16_1]], v[[B_F16]], v[[C_F16_1]]

; GCN: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; GCN: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_HI]], v[[R_F16_0]]
; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fma_v2f16_imm_b(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %c) {
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %r.val = call <2 x half> @llvm.fma.v2f16(<2 x half> %a.val, <2 x half> <half 3.0, half 3.0>, <2 x half> %c.val)
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fma_v2f16_imm_c:
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_mov_b32_e32 v[[C_F32:[0-9]+]], 0x40400000{{$}}
; VI:  v_mov_b32_e32 v[[C_F16:[0-9]+]], 0x4200{{$}}

; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]

; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_fma_f32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], v[[B_F32_0]], v[[C_F32]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_fma_f32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], v[[B_F32_1]], v[[C_F32]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]

; VI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; VI: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; VI:  v_fma_f16 v[[R_F16_0:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]], v[[C_F16]]
; VI:  v_fma_f16 v[[R_F16_1:[0-9]+]], v[[A_F16_1]], v[[B_F16_1]], v[[C_F16]]

; GCN: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; GCN: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_HI]], v[[R_F16_0]]
; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fma_v2f16_imm_c(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = call <2 x half> @llvm.fma.v2f16(<2 x half> %a.val, <2 x half> %b.val, <2 x half> <half 3.0, half 3.0>)
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}
