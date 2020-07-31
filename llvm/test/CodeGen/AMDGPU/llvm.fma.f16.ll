; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI,SIVI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,SIVI,VIGFX9 %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,VIGFX9 %s

declare half @llvm.fma.f16(half %a, half %b, half %c)
declare <2 x half> @llvm.fma.v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)
declare <4 x half> @llvm.fma.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)

; GCN-LABEL: {{^}}fma_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_fma_f32 v[[R_F32:[0-9]+]], v[[A_F32:[0-9]]], v[[B_F32:[0-9]]], v[[C_F32:[0-9]]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VIGFX9:  v_fma_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], v[[C_F16]]
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

; SI:  s_mov_b32 s[[A_F32:[0-9]+]], 0x40400000{{$}}
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_fma_f32 v[[R_F32:[0-9]+]], v[[B_F32:[0-9]]], s[[A_F32:[0-9]]], v[[C_F32:[0-9]]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VIGFX9:  s_movk_i32 s[[A_F16:[0-9]+]], 0x4200{{$}}
; VIGFX9:  v_fma_f16 v[[R_F16:[0-9]+]], v[[B_F16]], s[[A_F16]], v[[C_F16]]
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
; SI:  s_mov_b32 s[[B_F32:[0-9]+]], 0x40400000{{$}}
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_fma_f32 v[[R_F32:[0-9]+]], v[[A_F32:[0-9]]], s[[B_F32:[0-9]]], v[[C_F32:[0-9]]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VIGFX9:  s_movk_i32 s[[B_F16:[0-9]+]], 0x4200{{$}}
; VIGFX9:  v_fma_f16 v[[R_F16:[0-9]+]], v[[A_F16]], s[[B_F16]], v[[C_F16]]
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
; SI:  s_mov_b32 s[[C_F32:[0-9]+]], 0x40400000{{$}}
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_fma_f32 v[[R_F32:[0-9]+]], v[[A_F32:[0-9]]], v[[B_F32:[0-9]]], s[[C_F32:[0-9]]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VIGFX9:  s_movk_i32 s[[C_F16:[0-9]+]], 0x4200{{$}}
; VIGFX9:  v_fma_f16 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], s[[C_F16]]
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
; SI: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI: v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI: v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI: v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_F16_1]]

; SI: v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI: v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V2_F16]]


; SI-DAG:  v_fma_f32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], v[[B_F32_0]], v[[C_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_fma_f32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], v[[B_F32_1]], v[[C_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]

; VI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; VI: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; VI: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; VI-DAG: v_fma_f16 v[[R_F16_0:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]], v[[C_V2_F16]]
; VI-DAG: v_fma_f16 v[[R_F16_1:[0-9]+]], v[[A_F16_1]], v[[B_F16_1]], v[[C_F16_1]]

; GFX9: v_pk_fma_f16 v[[R_V2_F16:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]], v[[C_V2_F16]]

; SIVI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; GCN-NOT: and
; SIVI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]
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
; SI: buffer_load_dword v[[C_V2_F16:[0-9]+]]
; SI: buffer_load_dword v[[B_V2_F16:[0-9]+]]


; VIGFX9: buffer_load_dword v[[C_V2_F16:[0-9]+]]
; VIGFX9: buffer_load_dword v[[B_V2_F16:[0-9]+]]


; SI:  s_mov_b32 s[[A_F32:[0-9]+]], 0x40400000{{$}}
; VIGFX9:  s_movk_i32 s[[A_F16:[0-9]+]], 0x4200{{$}}
; SIVI-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SIVI-DAG: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]

; SI: v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_F16_1]]
; SI: v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI: v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V2_F16]]
; SI: v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]

; SI:  v_fma_f32 v[[R_F32_1:[0-9]+]], v[[B_F32_1]], s[[A_F32]], v[[C_F32_1]]
; SI-DAG:  v_fma_f32 v[[R_F32_0:[0-9]+]], v[[B_F32_0]], s[[A_F32]], v[[C_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]

; VI-DAG:  v_fma_f16 v[[R_F16_1:[0-9]+]], v[[B_F16_1]], s[[A_F16]], v[[C_F16_1]]
; VI-DAG:  v_fma_f16 v[[R_F16_0:[0-9]+]], v[[B_V2_F16]], s[[A_F16]], v[[C_V2_F16]]

; GFX9: v_pk_fma_f16 v[[R_V2_F16:[0-9]+]], v[[C_V2_F16]], s[[A_F16]], v[[B_V2_F16]]

; SIVI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; GCN-NOT: and
; SIVI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]
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
; SI: buffer_load_dword v[[C_V2_F16:[0-9]+]]
; SI: buffer_load_dword v[[A_V2_F16:[0-9]+]]

; VI:      buffer_load_dword v[[C_V2_F16:[0-9]+]]
; VIGFX9:  buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GFX9:    buffer_load_dword v[[C_V2_F16:[0-9]+]]

; SI:  s_mov_b32 s[[B_F32:[0-9]+]], 0x40400000{{$}}
; VIGFX9:  s_movk_i32 s[[B_F16:[0-9]+]], 0x4200{{$}}

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI-DAG:  v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V2_F16]]
; SI-DAG:  v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_F16_1]]
; SI-DAG:  v_fma_f32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], s[[B_F32]], v[[C_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_fma_f32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], s[[B_F32]], v[[C_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]

; VI-DAG:  v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; VI-DAG:  v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; VI-DAG:  v_fma_f16 v[[R_F16_0:[0-9]+]], v[[A_V2_F16]], s[[B_F16]], v[[C_V2_F16]]
; VI-DAG:  v_fma_f16 v[[R_F16_1:[0-9]+]], v[[A_F16_1]], s[[B_F16]], v[[C_F16_1]]

; GFX9: v_pk_fma_f16 v[[R_V2_F16:[0-9]+]], v[[A_V2_F16]], s[[B_F16]], v[[C_V2_F16]]

; SIVI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; GCN-NOT: and
; SIVI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]
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
; SI: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI: buffer_load_dword v[[A_V2_F16:[0-9]+]]

; GFX9:   buffer_load_dword v[[A_V2_F16:[0-9]+]]
; VIGFX9: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; VI:     buffer_load_dword v[[A_V2_F16:[0-9]+]]

; SI:  s_mov_b32 s[[C_F32:[0-9]+]], 0x40400000{{$}}
; VIGFX9:  s_movk_i32 s[[C_F16:[0-9]+]], 0x4200{{$}}

; SI:  v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]

; SI:  v_fma_f32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], v[[B_F32_1]], s[[C_F32]]
; SI-DAG:  v_fma_f32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], v[[B_F32_0]], s[[C_F32]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI:  v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; GCN-NOT: and
; SI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; VI-DAG:  v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; VI-DAG:  v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; VI-DAG:  v_fma_f16 v[[R_F16_0:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]], s[[C_F16]]
; VI-DAG:  v_fma_f16 v[[R_F16_1:[0-9]+]], v[[A_F16_1]], v[[B_F16_1]], s[[C_F16]]
; GCN-NOT: and
; VI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_1]]

; GFX9: v_pk_fma_f16 v[[R_V2_F16:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]], s[[C_F16]]

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

; GCN-LABEL: {{^}}fma_v4f16
; GCN: buffer_load_dwordx2 v{{\[}}[[A_V4_F16_LO:[0-9]+]]:[[A_V4_F16_HI:[0-9]+]]{{\]}}
; GCN: buffer_load_dwordx2 v{{\[}}[[B_V4_F16_LO:[0-9]+]]:[[B_V4_F16_HI:[0-9]+]]{{\]}}
; GCN: buffer_load_dwordx2 v{{\[}}[[C_V4_F16_LO:[0-9]+]]:[[C_V4_F16_HI:[0-9]+]]{{\]}}

; SI-DAG: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V4_F16_LO]]
; SI-DAG: v_lshrrev_b32_e32 v[[A_F16_0:[0-9]+]], 16, v[[A_V4_F16_LO]]
; SI-DAG: v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_V4_F16_HI]]
; SI-DAG: v_lshrrev_b32_e32 v[[A_F16_2:[0-9]+]], 16, v[[A_V4_F16_HI]]
; SI-DAG: v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V4_F16_LO]]
; SI-DAG: v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V4_F16_LO]]
; SI-DAG: v_lshrrev_b32_e32 v[[B_F16_0:[0-9]+]], 16, v[[B_V4_F16_LO]]
; SI-DAG: v_lshrrev_b32_e32 v[[C_F16_0:[0-9]+]], 16, v[[C_V4_F16_LO]]
; SI-DAG: v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_V4_F16_HI]]
; SI-DAG: v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_V4_F16_HI]]
; SI-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V4_F16_HI]]
; SI-DAG: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V4_F16_HI]]
; SI-DAG: v_cvt_f32_f16_e32 v[[A_F32_2:[0-9]+]], v[[A_V4_F16_LO]]
; SI-DAG: v_cvt_f32_f16_e32 v[[A_F32_3:[0-9]+]], v[[A_V4_F16_HI]]
; SI-DAG: v_cvt_f32_f16_e32 v[[B_F32_2:[0-9]+]], v[[B_V4_F16_LO]]
; SI-DAG: v_cvt_f32_f16_e32 v[[B_F32_3:[0-9]+]], v[[B_V4_F16_HI]]
; SI-DAG: v_cvt_f32_f16_e32 v[[C_F32_2:[0-9]+]], v[[C_V4_F16_LO]]
; SI-DAG: v_cvt_f32_f16_e32 v[[C_F32_3:[0-9]+]], v[[C_V4_F16_HI]]

; SI-DAG: v_fma_f32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], v[[B_F32_0]], v[[C_F32_0]]
; SI-DAG: v_fma_f32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], v[[B_F32_1]], v[[C_F32_1]]
; SI-DAG: v_fma_f32 v[[R_F32_2:[0-9]+]], v[[A_F32_2]], v[[B_F32_2]], v[[C_F32_2]]
; SI-DAG: v_fma_f32 v[[R_F32_3:[0-9]+]], v[[A_F32_3]], v[[B_F32_3]], v[[C_F32_3]]

; SI-DAG: v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG: v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI-DAG: v_cvt_f16_f32_e32 v[[R_F16_2:[0-9]+]], v[[R_F32_2]]
; SI-DAG: v_cvt_f16_f32_e32 v[[R_F16_3:[0-9]+]], v[[R_F32_3]]

; SI-DAG: v_lshlrev_b32_e32 v[[R1_F16_0:[0-9]]], 16, v[[R_F16_2]]
; SI-DAG: v_lshlrev_b32_e32 v[[R1_F16_1:[0-9]]], 16, v[[R_F16_3]]

; VI-DAG: v_lshrrev_b32_e32 v[[A_F16_0:[0-9]+]], 16, v[[A_V4_F16_LO]]
; VI-DAG: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V4_F16_HI]]
; VI-DAG: v_lshrrev_b32_e32 v[[B_F16_0:[0-9]+]], 16, v[[B_V4_F16_LO]]
; VI-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V4_F16_HI]]
; VI-DAG: v_lshrrev_b32_e32 v[[C_F16_0:[0-9]+]], 16, v[[C_V4_F16_LO]]
; VI-DAG: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V4_F16_HI]]

; VI-DAG: v_fma_f16 v[[R_F16_0:[0-9]+]], v[[A_V4_F16_LO]], v[[B_V4_F16_LO]], v[[C_V4_F16_LO]]
; VI-DAG: v_fma_f16 v[[R1_F16_0:[0-9]+]], v[[A_F16_0]], v[[B_F16_0]], v[[C_F16_0]]
; VI-DAG: v_fma_f16 v[[R_F16_1:[0-9]+]], v[[A_V4_F16_HI]], v[[B_V4_F16_HI]], v[[C_V4_F16_HI]]
; VI-DAG: v_fma_f16 v[[R1_F16_1:[0-9]+]], v[[A_F16_1]], v[[B_F16_1]], v[[C_F16_1]]

; SIVI-DAG: v_or_b32_e32 v[[R_V4_F16_LO:[0-9]+]], v[[R_F16_0]], v[[R1_F16_0]]
; SIVI-DAG: v_or_b32_e32 v[[R_V4_F16_HI:[0-9]+]], v[[R_F16_1]], v[[R1_F16_1]]

; GFX9-DAG: v_pk_fma_f16 v[[R_V4_F16_LO:[0-9]+]], v[[A_V4_F16_LO]], v[[B_V4_F16_LO]], v[[C_V4_F16_LO]]
; GFX9-DAG: v_pk_fma_f16 v[[R_V4_F16_HI:[0-9]+]], v[[A_V4_F16_HI]], v[[B_V4_F16_HI]], v[[C_V4_F16_HI]]

; GCN: buffer_store_dwordx2 v{{\[}}[[R_V4_F16_LO]]:[[R_V4_F16_HI]]{{\]}}
; GCN: s_endpgm

define amdgpu_kernel void @fma_v4f16(
    <4 x half> addrspace(1)* %r,
    <4 x half> addrspace(1)* %a,
    <4 x half> addrspace(1)* %b,
    <4 x half> addrspace(1)* %c) {
  %a.val = load <4 x half>, <4 x half> addrspace(1)* %a
  %b.val = load <4 x half>, <4 x half> addrspace(1)* %b
  %c.val = load <4 x half>, <4 x half> addrspace(1)* %c
  %r.val = call <4 x half> @llvm.fma.v4f16(<4 x half> %a.val, <4 x half> %b.val, <4 x half> %c.val)
  store <4 x half> %r.val, <4 x half> addrspace(1)* %r
  ret void
}
