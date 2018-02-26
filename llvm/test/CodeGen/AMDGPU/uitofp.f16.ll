; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}uitofp_i16_to_f16
; GCN: buffer_load_ushort v[[A_I16:[0-9]+]]
; SI:  v_cvt_f32_u32_e32 v[[A_F32:[0-9]+]], v[[A_I16]]
; VI:  v_cvt_f32_i32_e32 v[[A_F32:[0-9]+]], v[[A_I16]]
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_F32]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @uitofp_i16_to_f16(
    half addrspace(1)* %r,
    i16 addrspace(1)* %a) {
entry:
  %a.val = load i16, i16 addrspace(1)* %a
  %r.val = uitofp i16 %a.val to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}uitofp_i32_to_f16
; GCN: buffer_load_dword v[[A_I32:[0-9]+]]
; GCN: v_cvt_f32_u32_e32 v[[A_I16:[0-9]+]], v[[A_I32]]
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_I16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @uitofp_i32_to_f16(
    half addrspace(1)* %r,
    i32 addrspace(1)* %a) {
entry:
  %a.val = load i32, i32 addrspace(1)* %a
  %r.val = uitofp i32 %a.val to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; f16 = uitofp i64 is in uint_to_fp.i64.ll

; GCN-LABEL: {{^}}uitofp_v2i16_to_v2f16
; GCN:     buffer_load_dword

; SI: v_cvt_f32_u32_e32
; SI: v_cvt_f32_u32_e32
; SI: v_cvt_f16_f32_e32
; SI: v_cvt_f16_f32_e32
; SI-DAG: v_lshlrev_b32_e32
; SI: v_or_b32_e32

; VI-DAG: v_cvt_f16_f32_e32
; VI-DAG: v_cvt_f32_i32_sdwa
; VI-DAG: v_cvt_f32_i32_sdwa
; VI-DAG: v_cvt_f16_f32_sdwa
; VI:     v_or_b32_e32

; GCN: buffer_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @uitofp_v2i16_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %r.val = uitofp <2 x i16> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}uitofp_v2i32_to_v2f16
; GCN:     buffer_load_dwordx2

; SI: v_cvt_f32_u32_e32
; SI: v_cvt_f32_u32_e32
; SI: v_cvt_f16_f32_e32
; SI: v_cvt_f16_f32_e32
; SI-DAG: v_lshlrev_b32_e32
; SI: v_or_b32_e32

; VI-DAG: v_cvt_f32_u32_e32
; VI-DAG: v_cvt_f32_u32_e32
; VI-DAG: v_cvt_f16_f32_e32
; VI-DAG: v_cvt_f16_f32_sdwa
; VI:     v_or_b32_e32

; GCN:     buffer_store_dword
; GCN:     s_endpgm
define amdgpu_kernel void @uitofp_v2i32_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x i32> addrspace(1)* %a) {
entry:
  %a.val = load <2 x i32>, <2 x i32> addrspace(1)* %a
  %r.val = uitofp <2 x i32> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; f16 = uitofp i64 is in uint_to_fp.i64.ll
