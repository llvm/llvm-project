; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -denormal-fp-math=preserve-sign -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -denormal-fp-math=preserve-sign -denormal-fp-math-f32=preserve-sign -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,VI-FLUSH %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -denormal-fp-math=ieee -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -denormal-fp-math=ieee -denormal-fp-math-f32=preserve-sign -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,VI-DENORM %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx1010 -denormal-fp-math=preserve-sign -denormal-fp-math-f32=preserve-sign -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX10,GFX10-FLUSH %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx1010 -denormal-fp-math=ieee -denormal-fp-math-f32=preserve-sign -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX10,GFX10-DENORM %s

declare half @llvm.fmuladd.f16(half %a, half %b, half %c)
declare <2 x half> @llvm.fmuladd.v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)

; GCN-LABEL: {{^}}fmuladd_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_mac_f32_e32 v[[C_F32]], v[[A_F32]], v[[B_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[C_F32]]
; SI:  buffer_store_short v[[R_F16]]

; VI-FLUSH: v_mac_f16_e32 v[[C_F16]], v[[A_F16]], v[[B_F16]]
; VI-FLUSH: buffer_store_short v[[C_F16]]

; VI-DENORM: v_fma_f16 [[RESULT:v[0-9]+]], v[[A_F16]], v[[B_F16]], v[[C_F16]]
; VI-DENORM: buffer_store_short [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e32 [[MUL:v[0-9]+]], v[[A_F16]], v[[B_F16]]
; GFX10-FLUSH: v_add_f16_e32 [[ADD:v[0-9]+]], [[MUL]], v[[C_F16]]
; GFX10-FLUSH: buffer_store_short [[ADD]]

; GFX10-DENORM: v_fmac_f16_e32 v[[C_F16]], v[[A_F16]], v[[B_F16]]
; GFX10-DENORM: buffer_store_short v[[C_F16]],

; GCN: s_endpgm
define amdgpu_kernel void @fmuladd_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %r.val = call half @llvm.fmuladd.f16(half %a.val, half %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fmuladd_f16_imm_a
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_mac_f32_e32 v[[C_F32]], 0x40400000, v[[B_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[C_F32]]
; SI:  buffer_store_short v[[R_F16]]

; VI-FLUSH: v_mac_f16_e32 v[[C_F16]], 0x4200, v[[B_F16]]
; VI-FLUSH: buffer_store_short v[[C_F16]]

; VI-DENORM: s_movk_i32 [[KA:s[0-9]+]], 0x4200
; VI-DENORM: v_fma_f16 [[RESULT:v[0-9]+]], v[[B_F16]], [[KA]], v[[C_F16]]
; VI-DENORM: buffer_store_short [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e32 [[MUL:v[0-9]+]], 0x4200, v[[B_F16]]
; GFX10-FLUSH: v_add_f16_e32 [[ADD:v[0-9]+]], [[MUL]], v[[C_F16]]
; GFX10-FLUSH: buffer_store_short [[ADD]]

; GFX10-DENORM: v_fmac_f16_e32 v[[C_F16]], 0x4200, v[[B_F16]]
; GFX10-DENORM: buffer_store_short v[[C_F16]],

; GCN: s_endpgm
define amdgpu_kernel void @fmuladd_f16_imm_a(
    half addrspace(1)* %r,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %b.val = load volatile half, half addrspace(1)* %b
  %c.val = load volatile half, half addrspace(1)* %c
  %r.val = call half @llvm.fmuladd.f16(half 3.0, half %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fmuladd_f16_imm_b
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_mac_f32_e32 v[[C_F32]], 0x40400000, v[[A_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[C_F32]]
; SI:  buffer_store_short v[[R_F16]]

; VI-FLUSH: v_mac_f16_e32 v[[C_F16]], 0x4200, v[[A_F16]]
; VI-FLUSH: buffer_store_short v[[C_F16]]

; VI-DENORM: s_movk_i32 [[KA:s[0-9]+]], 0x4200
; VI-DENORM: v_fma_f16 [[RESULT:v[0-9]+]], v[[A_F16]], [[KA]], v[[C_F16]]
; VI-DENORM: buffer_store_short [[RESULT]]

; GFX10-FLUSH: v_mul_f16_e32 [[MUL:v[0-9]+]], 0x4200, v[[A_F16]]
; GFX10-FLUSH: v_add_f16_e32 [[ADD:v[0-9]+]], [[MUL]], v[[C_F16]]
; GFX10-FLUSH: buffer_store_short [[ADD]]

; GFX10-DENORM: v_fmac_f16_e32 v[[C_F16]], 0x4200, v[[A_F16]]
; GFX10-DENORM: buffer_store_short v[[C_F16]],

; GCN: s_endpgm
define amdgpu_kernel void @fmuladd_f16_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %c) {
  %a.val = load volatile half, half addrspace(1)* %a
  %c.val = load volatile half, half addrspace(1)* %c
  %r.val = call half @llvm.fmuladd.f16(half %a.val, half 3.0, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fmuladd_v2f16
; SI: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; SI: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI: buffer_load_dword v[[C_V2_F16:[0-9]+]]

; VI-FLUSH: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; VI-FLUSH: buffer_load_dword v[[C_V2_F16:[0-9]+]]
; VI-FLUSH: buffer_load_dword v[[B_V2_F16:[0-9]+]]

; VI-DENORM: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; VI-DENORM: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; VI-DENORM: buffer_load_dword v[[C_V2_F16:[0-9]+]]

; GFX10: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GFX10: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; GFX10: buffer_load_dword v[[C_V2_F16:[0-9]+]]

; SI: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]

; SI-DAG: v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI-DAG: v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V2_F16]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_F16_1]]
; SI-DAG:  v_mac_f32_e32 v[[C_F32_0]], v[[A_F32_0]], v[[B_F32_0]]
; SI-DAG:  v_mac_f32_e32 v[[C_F32_1]], v[[A_F32_1]], v[[B_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[C_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_LO:[0-9]+]], v[[C_F32_0]]
; SI-DAG:  v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI:  v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_LO]], v[[R_F16_HI]]

; VI-FLUSH:     v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; VI-FLUSH-DAG: v_mac_f16_sdwa v[[C_F16_1]], v[[A_V2_F16]], v[[B_V2_F16]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-FLUSH-DAG: v_mac_f16_e32 v[[C_V2_F16]], v[[A_V2_F16]], v[[B_V2_F16]]
; VI-FLUSH-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[C_F16_1]]
; VI-FLUSH-NOT: v_and_b32
; VI-FLUSH:     v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[C_V2_F16]], v[[R_F16_HI]]

; VI-DENORM-DAG: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; VI-DENORM-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; VI-DENORM-DAG: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; VI-DENORM-DAG: v_fma_f16 v[[RES0:[0-9]+]], v[[C_V2_F16]], v[[B_V2_F16]], v[[A_V2_F16]]
; VI-DENORM-DAG: v_fma_f16 v[[RES1:[0-9]+]], v[[C_F16_1]], v[[B_F16_1]], v[[A_F16_1]]
; VI-DENORM-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[RES1]]
; VI-DENORM-NOT: v_and_b32
; VI-DENORM: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[RES0]], v[[R_F16_HI]]

; GFX10-FLUSH: v_pk_mul_f16 [[MUL:v[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]]
; GFX10-FLUSH: v_pk_add_f16 v[[R_V2_F16:[0-9]+]], [[MUL]], v[[C_V2_F16]]

; GFX10-DENORM: v_pk_fma_f16 v[[R_V2_F16:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]], v[[C_V2_F16]]

; GCN: buffer_store_dword v[[R_V2_F16]]
define amdgpu_kernel void @fmuladd_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) {
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %r.val = call <2 x half> @llvm.fmuladd.v2f16(<2 x half> %a.val, <2 x half> %b.val, <2 x half> %c.val)
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}
