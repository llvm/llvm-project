; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=SIVI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX89 -check-prefix=VI -check-prefix=SIVI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX89 -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}fsub_f16:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_sub_f32_e32 v[[R_F32:[0-9]+]], v[[A_F32]], v[[B_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; GFX89:  v_sub_f16_e32 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fsub_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load volatile half, half addrspace(1)* %a
  %b.val = load volatile half, half addrspace(1)* %b
  %r.val = fsub half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fsub_f16_imm_a:
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_sub_f32_e32 v[[R_F32:[0-9]+]], 1.0, v[[B_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; GFX89:  v_sub_f16_e32 v[[R_F16:[0-9]+]], 1.0, v[[B_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fsub_f16_imm_a(
    half addrspace(1)* %r,
    half addrspace(1)* %b) {
entry:
  %b.val = load volatile half, half addrspace(1)* %b
  %r.val = fsub half 1.0, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fsub_f16_imm_b:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_add_f32_e32 v[[R_F32:[0-9]+]], -2.0, v[[A_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; GFX89:  v_add_f16_e32 v[[R_F16:[0-9]+]], -2.0, v[[A_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fsub_f16_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load volatile half, half addrspace(1)* %a
  %r.val = fsub half %a.val, 2.0
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fsub_v2f16:
; SI: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI: buffer_load_dword v[[A_V2_F16:[0-9]+]]

; SI-DAG: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI-DAG: v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI-DAG:  v_sub_f32_e32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], v[[B_F32_0]]
; SI-DAG:  v_sub_f32_e32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], v[[B_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; VI: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; VI: buffer_load_dword v[[A_V2_F16:[0-9]+]]

; VI-DAG: v_sub_f16_e32 v[[R_F16_0:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]]
; VI-DAG: v_sub_f16_sdwa v[[R_F16_HI:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]


; GFX9: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GFX9: buffer_load_dword v[[B_V2_F16:[0-9]+]]

; GFX9: v_pk_add_f16 v[[R_V2_F16:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]] neg_lo:[0,1] neg_hi:[0,1]

; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm

define amdgpu_kernel void @fsub_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fsub <2 x half> %a.val, %b.val
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fsub_v2f16_imm_a:
; GCN-DAG: buffer_load_dword v[[B_V2_F16:[0-9]+]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI-DAG: v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI-DAG: v_sub_f32_e32 v[[R_F32_0:[0-9]+]], 1.0, v[[B_F32_0]]
; SI-DAG: v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG: v_sub_f32_e32 v[[R_F32_1:[0-9]+]], 2.0, v[[B_F32_1]]
; SI-DAG: v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; VI-DAG: v_mov_b32_e32 [[CONST2:v[0-9]+]], 0x4000
; VI-DAG: v_sub_f16_sdwa v[[R_F16_HI:[0-9]+]], [[CONST2]], v[[B_V2_F16]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; VI-DAG: v_sub_f16_e32 v[[R_F16_0:[0-9]+]], 1.0, v[[B_V2_F16]]
; VI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; GFX9: s_mov_b32 [[K:s[0-9]+]], 0x40003c00
; GFX9: v_pk_add_f16 v[[R_V2_F16:[0-9]+]], v[[B_V2_F16]], [[K]] neg_lo:[1,0] neg_hi:[1,0]

; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm

define amdgpu_kernel void @fsub_v2f16_imm_a(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %b) {
entry:
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fsub <2 x half> <half 1.0, half 2.0>, %b.val
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fsub_v2f16_imm_b:
; GCN-DAG: buffer_load_dword v[[A_V2_F16:[0-9]+]]

; SI-DAG: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI-DAG: v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI-DAG: v_add_f32_e32 v[[R_F32_0:[0-9]+]], -2.0, v[[A_F32_0]]
; SI-DAG: v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG: v_add_f32_e32 v[[R_F32_1:[0-9]+]], -1.0, v[[A_F32_1]]
; SI-DAG: v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; VI-DAG: v_mov_b32_e32 [[CONSTM1:v[0-9]+]], 0xbc00
; VI-DAG: v_add_f16_sdwa v[[R_F16_HI:[0-9]+]], v[[A_V2_F16]], [[CONSTM1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_add_f16_e32 v[[R_F16_0:[0-9]+]], -2.0, v[[A_V2_F16]]
; VI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; GFX9: s_mov_b32 [[K:s[0-9]+]], 0xbc00c000
; GFX9: v_pk_add_f16 v[[R_V2_F16:[0-9]+]], v[[A_V2_F16]], [[K]]{{$}}

; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm

define amdgpu_kernel void @fsub_v2f16_imm_b(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %r.val = fsub <2 x half> %a.val, <half 2.0, half 1.0>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}
