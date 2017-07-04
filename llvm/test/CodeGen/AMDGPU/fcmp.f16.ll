; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}fcmp_f16_lt
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_lt_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_lt_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_lt(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp olt half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_lt_abs:
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]

; SI:  v_cvt_f32_f16_e64 v[[A_F32:[0-9]+]], |v[[A_F16]]|
; SI:  v_cvt_f32_f16_e64 v[[B_F32:[0-9]+]], |v[[B_F16]]|

; SI:  v_cmp_lt_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_lt_f16_e64 s{{\[[0-9]+:[0-9]+\]}}, |v[[A_F16]]|, |v[[B_F16]]|

; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_lt_abs(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %a.abs = call half @llvm.fabs.f16(half %a.val)
  %b.abs = call half @llvm.fabs.f16(half %b.val)
  %r.val = fcmp olt half %a.abs, %b.abs
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_eq
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_eq_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_eq_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_eq(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp oeq half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_le
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_le_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_le_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_le(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp ole half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_gt
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_gt_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_gt_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_gt(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp ogt half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_lg
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_lg_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_lg_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_lg(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp one half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_ge
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_ge_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_ge_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_ge(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp oge half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_o
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_o_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_o_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_o(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp ord half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_u
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_u_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_u_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_u(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp uno half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_nge
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_nge_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_nge_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_nge(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp ult half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_nlg
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_nlg_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_nlg_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_nlg(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp ueq half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_ngt
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_ngt_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_ngt_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_ngt(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp ule half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_nle
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_nle_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_nle_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_nle(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp ugt half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_neq
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_neq_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_neq_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_neq(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp une half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_f16_nlt
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cmp_nlt_f32_e32 vcc, v[[A_F32]], v[[B_F32]]
; VI:  v_cmp_nlt_f16_e32 vcc, v[[A_F16]], v[[B_F16]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_f16_nlt(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fcmp uge half %a.val, %b.val
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_lt
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_lt_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_lt_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_lt_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_lt_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_lt(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp olt <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_eq
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_eq_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_eq_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_eq_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_eq_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_eq(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp oeq <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_le
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_le_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_le_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_le_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_le_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_le(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp ole <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_gt
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_gt_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_gt_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_gt_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_gt_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_gt(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp ogt <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_lg
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_lg_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_lg_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_lg_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_lg_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_lg(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp one <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_ge
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_ge_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_ge_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_ge_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_ge_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_ge(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp oge <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_o
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_o_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_o_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_o_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_o_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_o(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp ord <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_u
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_u_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_u_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_u_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_u_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_u(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp uno <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_nge
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_nge_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_nge_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_nge_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_nge_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_nge(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp ult <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_nlg
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_nlg_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_nlg_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_nlg_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_nlg_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_nlg(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp ueq <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_ngt
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_ngt_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_ngt_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_ngt_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_ngt_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_ngt(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp ule <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_nle
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_nle_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_nle_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_nle_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_nle_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_nle(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp ugt <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_neq
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_neq_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_neq_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_neq_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_neq_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_neq(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp une <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fcmp_v2f16_nlt
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; GCN: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cmp_nlt_f32_e32 vcc, v[[A_F32_0]], v[[B_F32_0]]
; SI:  v_cmp_nlt_f32_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F32_1]], v[[B_F32_1]]
; VI:  v_cmp_nlt_f16_e32 vcc, v[[A_V2_F16]], v[[B_V2_F16]]
; VI:  v_cmp_nlt_f16_e64 s[{{[0-9]+}}:{{[0-9]+}}], v[[A_F16_1]], v[[B_F16_1]]
; GCN: v_cndmask_b32_e64 v[[R_I32_0:[0-9]+]]
; GCN: v_cndmask_b32_e64 v[[R_I32_1:[0-9]+]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_I32_0]]:[[R_I32_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fcmp_v2f16_nlt(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %r.val = fcmp uge <2 x half> %a.val, %b.val
  %r.val.sext = sext <2 x i1> %r.val to <2 x i32>
  store <2 x i32> %r.val.sext, <2 x i32> addrspace(1)* %r
  ret void
}

declare half @llvm.fabs.f16(half) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
