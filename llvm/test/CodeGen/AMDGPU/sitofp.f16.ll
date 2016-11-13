; RUN: llc -march=amdgcn -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}sitofp_i16_to_f16
; GCN: buffer_load_{{sshort|ushort}} v[[A_I16:[0-9]+]]
; SI:  v_cvt_f32_i32_e32 v[[A_F32:[0-9]+]], v[[A_I16]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_F32]]
; VI:  v_cvt_f16_i16_e32 v[[R_F16:[0-9]+]], v[[A_I16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define void @sitofp_i16_to_f16(
    half addrspace(1)* %r,
    i16 addrspace(1)* %a) {
entry:
  %a.val = load i16, i16 addrspace(1)* %a
  %r.val = sitofp i16 %a.val to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}sitofp_i32_to_f16
; GCN: buffer_load_dword v[[A_I32:[0-9]+]]
; GCN: v_cvt_f32_i32_e32 v[[A_I16:[0-9]+]], v[[A_I32]]
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_I16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define void @sitofp_i32_to_f16(
    half addrspace(1)* %r,
    i32 addrspace(1)* %a) {
entry:
  %a.val = load i32, i32 addrspace(1)* %a
  %r.val = sitofp i32 %a.val to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; f16 = sitofp i64 is in sint_to_fp.i64.ll

; GCN-LABEL: {{^}}sitofp_v2i16_to_v2f16
; GCN: buffer_load_dword v[[A_V2_I16:[0-9]+]]
; SI:  v_bfe_i32 v[[A_I16_0:[0-9]+]], v[[A_V2_I16]], 0, 16
; SI:  v_ashrrev_i32_e32 v[[A_I16_1:[0-9]+]], 16, v[[A_V2_I16]]
; SI:  v_cvt_f32_i32_e32 v[[A_F32_1:[0-9]+]], v[[A_I16_1]]
; SI:  v_cvt_f32_i32_e32 v[[A_F32_0:[0-9]+]], v[[A_I16_0]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[A_F32_1]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[A_F32_0]]
; VI:  v_lshrrev_b32_e32 v[[A_I16_1:[0-9]+]], 16, v[[A_V2_I16]]
; VI:  v_cvt_f16_i16_e32 v[[R_F16_0:[0-9]+]], v[[A_V2_I16]]
; VI:  v_cvt_f16_i16_e32 v[[R_F16_1:[0-9]+]], v[[A_I16_1]]
; VI:  v_and_b32_e32 v[[R_F16_LO:[0-9]+]], 0xffff, v[[R_F16_0]]
; GCN: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI:  v_and_b32_e32 v[[R_F16_LO:[0-9]+]], 0xffff, v[[R_F16_0]]
; GCN: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_HI]], v[[R_F16_LO]]
; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define void @sitofp_v2i16_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %r.val = sitofp <2 x i16> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}sitofp_v2i32_to_v2f16
; GCN:     buffer_load_dwordx2
; GCN:     v_cvt_f32_i32_e32
; GCN:     v_cvt_f32_i32_e32
; GCN:     v_cvt_f16_f32_e32
; GCN:     v_cvt_f16_f32_e32
; GCN-DAG: v_and_b32_e32
; GCN-DAG: v_lshlrev_b32_e32
; GCN-DAG: v_or_b32_e32
; GCN:     buffer_store_dword
; GCN:     s_endpgm
define void @sitofp_v2i32_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x i32> addrspace(1)* %a) {
entry:
  %a.val = load <2 x i32>, <2 x i32> addrspace(1)* %a
  %r.val = sitofp <2 x i32> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; v2f16 = sitofp v2i64 is in sint_to_fp.i64.ll
