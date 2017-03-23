; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}br_cc_f16:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_nlt_f32_e32 vcc, v[[B_F32]], v[[A_F32]]
; VI:  v_cmp_nlt_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: s_cbranch_vccnz

; GCN: one{{$}}
; SI: v_cvt_f16_f32_e32 v[[A_F16:[0-9]+]], v[[B_F32]]
; GCN: buffer_store_short
; GCN: s_endpgm

; GCN: two{{$}}
; SI:  v_cvt_f16_f32_e32 v[[B_F16:[0-9]+]], v[[A_F32]]
; GCN: buffer_store_short v[[B_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @br_cc_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %fcmp = fcmp olt half %a.val, %b.val
  br i1 %fcmp, label %one, label %two

one:
  store half %a.val, half addrspace(1)* %r
  ret void

two:
  store half %b.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}br_cc_f16_imm_a:
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]

; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_nlt_f32_e32 vcc, 0.5, v[[B_F32]]
; SI: s_cbranch_vccnz

; VI:  v_cmp_nlt_f16_e32 vcc, 0.5, v[[B_F16]]
; VI: s_cbranch_vccnz

; GCN: one{{$}}
; VI: v_mov_b32_e32 v[[A_F16:[0-9]+]], 0x380{{0|1}}{{$}}

; SI: buffer_store_short v[[A_F16]]
; SI: s_endpgm


; GCN: two{{$}}
; SI:  v_cvt_f16_f32_e32 v[[B_F16:[0-9]+]], v[[B_F32]]

define amdgpu_kernel void @br_cc_f16_imm_a(
    half addrspace(1)* %r,
    half addrspace(1)* %b) {
entry:
  %b.val = load half, half addrspace(1)* %b
  %fcmp = fcmp olt half 0xH3800, %b.val
  br i1 %fcmp, label %one, label %two

one:
  store half 0xH3800, half addrspace(1)* %r
  ret void

two:
  store half %b.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}br_cc_f16_imm_b:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]

; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cmp_ngt_f32_e32 vcc, 0.5, v[[A_F32]]

; VI:  v_cmp_ngt_f16_e32 vcc, 0.5, v[[A_F16]]
; GCN: s_cbranch_vccnz

; GCN: one{{$}}
; SI:  v_cvt_f16_f32_e32 v[[A_F16:[0-9]+]], v[[A_F32]]

; GCN: two{{$}}
; VI:  v_mov_b32_e32 v[[B_F16:[0-9]+]], 0x3800{{$}}
; GCN: buffer_store_short v[[B_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @br_cc_f16_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %fcmp = fcmp olt half %a.val, 0xH3800
  br i1 %fcmp, label %one, label %two

one:
  store half %a.val, half addrspace(1)* %r
  ret void

two:
  store half 0xH3800, half addrspace(1)* %r
  ret void
}
