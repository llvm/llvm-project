; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}select_f16:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[D_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_lt_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_cvt_f32_f16_e32 v[[D_F32:[0-9]+]], v[[D_F16]]
; SI:  v_cndmask_b32_e32 v[[R_F32:[0-9]+]], v[[D_F32]], v[[C_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_cmp_lt_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; VI:  v_cndmask_b32_e32 v[[R_F16:[0-9]+]], v[[D_F16]], v[[C_F16]], vcc
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define void @select_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c,
    half addrspace(1)* %d) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %d.val = load half, half addrspace(1)* %d
  %fcmp = fcmp olt half %a.val, %b.val
  %r.val = select i1 %fcmp, half %c.val, half %d.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_f16_imm_a:
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[D_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_lt_f32_e32 vcc, 0.5, v[[B_F32]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_cvt_f32_f16_e32 v[[D_F32:[0-9]+]], v[[D_F16]]
; SI:  v_cndmask_b32_e32 v[[R_F32:[0-9]+]], v[[D_F32]], v[[C_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_cmp_lt_f16_e32 vcc, 0.5, v[[B_F16]]
; VI:  v_cndmask_b32_e32 v[[R_F16:[0-9]+]], v[[D_F16]], v[[C_F16]], vcc
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define void @select_f16_imm_a(
    half addrspace(1)* %r,
    half addrspace(1)* %b,
    half addrspace(1)* %c,
    half addrspace(1)* %d) {
entry:
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %d.val = load half, half addrspace(1)* %d
  %fcmp = fcmp olt half 0xH3800, %b.val
  %r.val = select i1 %fcmp, half %c.val, half %d.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_f16_imm_b:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[D_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cmp_gt_f32_e32 vcc, 0.5, v[[A_F32]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_cvt_f32_f16_e32 v[[D_F32:[0-9]+]], v[[D_F16]]
; SI:  v_cndmask_b32_e32 v[[R_F32:[0-9]+]], v[[D_F32]], v[[C_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]

; VI:  v_cmp_gt_f16_e32 vcc, 0.5, v[[A_F16]]
; VI:  v_cndmask_b32_e32 v[[R_F16:[0-9]+]], v[[D_F16]], v[[C_F16]], vcc
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define void @select_f16_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %c,
    half addrspace(1)* %d) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %c.val = load half, half addrspace(1)* %c
  %d.val = load half, half addrspace(1)* %d
  %fcmp = fcmp olt half %a.val, 0xH3800
  %r.val = select i1 %fcmp, half %c.val, half %d.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_f16_imm_c:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[D_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[D_F32:[0-9]+]], v[[D_F16]]
; SI:  v_cmp_nlt_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; SI:  v_cndmask_b32_e32 v[[R_F32:[0-9]+]], 0.5, v[[D_F32]], vcc
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]

; VI:  v_cmp_nlt_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; VI:  v_mov_b32_e32 v[[C_F16:[0-9]+]], 0x3800{{$}}
; VI:  v_cndmask_b32_e32 v[[R_F16:[0-9]+]], v[[C_F16]], v[[D_F16]], vcc
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define void @select_f16_imm_c(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %d) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %d.val = load half, half addrspace(1)* %d
  %fcmp = fcmp olt half %a.val, %b.val
  %r.val = select i1 %fcmp, half 0xH3800, half %d.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_f16_imm_d:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[C_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_cmp_lt_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; SI:  v_cndmask_b32_e32 v[[R_F32:[0-9]+]], 0.5, v[[C_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_cmp_lt_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; VI:  v_mov_b32_e32 v[[D_F16:[0-9]+]], 0x3800{{$}}
; VI:  v_cndmask_b32_e32 v[[R_F16:[0-9]+]], v[[D_F16]], v[[C_F16]], vcc
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define void @select_f16_imm_d(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %fcmp = fcmp olt half %a.val, %b.val
  %r.val = select i1 %fcmp, half %c.val, half 0xH3800
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_v2f16:
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cmp_lt_f32_e64
; SI:  v_cmp_lt_f32_e32
; VI:  v_cmp_lt_f16_e32
; VI:  v_cmp_lt_f16_e64
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e64
; SI:  v_cvt_f16_f32_e32
; SI:  v_cvt_f16_f32_e32
; GCN: s_endpgm
define void @select_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c,
    <2 x half> addrspace(1)* %d) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %d.val = load <2 x half>, <2 x half> addrspace(1)* %d
  %fcmp = fcmp olt <2 x half> %a.val, %b.val
  %r.val = select <2 x i1> %fcmp, <2 x half> %c.val, <2 x half> %d.val
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_v2f16_imm_a:
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cmp_lt_f32_e32 vcc, 0.5
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cmp_lt_f32_e64

; VI:  v_cmp_lt_f16_e32
; VI:  v_cmp_lt_f16_e64
; GCN: v_cndmask_b32_e32
; SI:  v_cvt_f16_f32_e32
; GCN: v_cndmask_b32_e64
; SI:  v_cvt_f16_f32_e32
; GCN: s_endpgm
define void @select_v2f16_imm_a(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c,
    <2 x half> addrspace(1)* %d) {
entry:
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %d.val = load <2 x half>, <2 x half> addrspace(1)* %d
  %fcmp = fcmp olt <2 x half> <half 0xH3800, half 0xH3900>, %b.val
  %r.val = select <2 x i1> %fcmp, <2 x half> %c.val, <2 x half> %d.val
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_v2f16_imm_b:
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cmp_gt_f32_e32 vcc, 0.5
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cmp_gt_f32_e64

; VI:  v_cmp_gt_f16_e32
; VI:  v_cmp_gt_f16_e64
; GCN: v_cndmask_b32_e32
; SI:  v_cvt_f16_f32_e32
; GCN: v_cndmask_b32_e64
; SI:  v_cvt_f16_f32_e32
; GCN: s_endpgm
define void @select_v2f16_imm_b(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %c,
    <2 x half> addrspace(1)* %d) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %d.val = load <2 x half>, <2 x half> addrspace(1)* %d
  %fcmp = fcmp olt <2 x half> %a.val, <half 0xH3800, half 0xH3900>
  %r.val = select <2 x i1> %fcmp, <2 x half> %c.val, <2 x half> %d.val
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_v2f16_imm_c:
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32

; SI: v_cmp_nlt_f32_e32
; SI: v_cndmask_b32_e32
; SI: v_cmp_nlt_f32_e32
; SI: v_cndmask_b32_e32

; VI: v_cmp_nlt_f16_e32
; VI: v_cndmask_b32_e32

; VI: v_cmp_nlt_f16_e32
; VI: v_cndmask_b32_e32

; SI:  v_cvt_f16_f32_e32
; SI:  v_cvt_f16_f32_e32
; GCN: s_endpgm
define void @select_v2f16_imm_c(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %d) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %d.val = load <2 x half>, <2 x half> addrspace(1)* %d
  %fcmp = fcmp olt <2 x half> %a.val, %b.val
  %r.val = select <2 x i1> %fcmp, <2 x half> <half 0xH3800, half 0xH3900>, <2 x half> %d.val
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}select_v2f16_imm_d:
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cvt_f32_f16_e32
; SI:  v_cmp_lt_f32_e32
; SI:  v_cmp_lt_f32_e64
; VI:  v_cmp_lt_f16_e32
; VI:  v_cmp_lt_f16_e64
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e64
; SI:  v_cvt_f16_f32_e32
; SI:  v_cvt_f16_f32_e32
; GCN: s_endpgm
define void @select_v2f16_imm_d(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %fcmp = fcmp olt <2 x half> %a.val, %b.val
  %r.val = select <2 x i1> %fcmp, <2 x half> %c.val, <2 x half> <half 0xH3800, half 0xH3900>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}
