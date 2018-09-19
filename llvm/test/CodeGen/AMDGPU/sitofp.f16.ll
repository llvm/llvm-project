; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}sitofp_i16_to_f16
; GCN: buffer_load_{{sshort|ushort}} v[[A_I16:[0-9]+]]
; GCN: v_cvt_f32_i32_e32 v[[A_F32:[0-9]+]], v[[A_I16]]
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_F32]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @sitofp_i16_to_f16(
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
define amdgpu_kernel void @sitofp_i32_to_f16(
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
; GCN:     buffer_load_dword

; SI: v_cvt_f32_i32_e32
; SI: v_cvt_f32_i32_e32
; SI: v_cvt_f16_f32_e32
; SI: v_cvt_f16_f32_e32
; SI-DAG: v_lshlrev_b32_e32
; SI: v_or_b32_e32

; VI-DAG: v_cvt_f32_i32_sdwa
; VI-DAG: v_cvt_f32_i32_sdwa
; VI-DAG: v_cvt_f16_f32_e32
; VI-DAG: v_cvt_f16_f32_sdwa
; VI:     v_or_b32_e32

; GCN: buffer_store_dword
; GCN: s_endpgm

define amdgpu_kernel void @sitofp_v2i16_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %r.val = sitofp <2 x i16> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}sitofp_v2i32_to_v2f16
; GCN:    buffer_load_dwordx2

; SI: v_cvt_f32_i32_e32
; SI: v_cvt_f32_i32_e32
; SI: v_cvt_f16_f32_e32
; SI: v_cvt_f16_f32_e32
; SI-DAG: v_lshlrev_b32_e32
; SI: v_or_b32_e32

; VI-DAG: v_cvt_f32_i32_e32
; VI-DAG: v_cvt_f32_i32_e32
; VI-DAG: v_cvt_f16_f32_e32
; VI-DAG: v_cvt_f16_f32_sdwa
; VI:     v_or_b32_e32

; GCN: buffer_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @sitofp_v2i32_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x i32> addrspace(1)* %a) {
entry:
  %a.val = load <2 x i32>, <2 x i32> addrspace(1)* %a
  %r.val = sitofp <2 x i32> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; FUNC-LABEL: {{^}}s_sint_to_fp_i1_to_f16:
; GCN-DAG: v_cmp_le_f32_e32 [[CMP0:vcc]], 1.0, {{v[0-9]+}}
; GCN-DAG: v_cmp_le_f32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], 0, {{v[0-9]+}}
; GCN: s_xor_b64 [[R_CMP:s\[[0-9]+:[0-9]+\]]], [[CMP1]], [[CMP0]]
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1.0, [[R_CMP]]
; GCN-NEXT: v_cvt_f16_f32_e32 [[R_F16:v[0-9]+]], [[RESULT]]
; GCN: buffer_store_short
; GCN: s_endpgm
define amdgpu_kernel void @s_sint_to_fp_i1_to_f16(half addrspace(1)* %out, float addrspace(1)* %in0, float addrspace(1)* %in1) {
  %a = load float, float addrspace(1) * %in0
  %b = load float, float addrspace(1) * %in1
  %acmp = fcmp oge float %a, 0.000000e+00
  %bcmp = fcmp oge float %b, 1.000000e+00
  %result = xor i1 %acmp, %bcmp
  %fp = sitofp i1 %result to half
  store half %fp, half addrspace(1)* %out
  ret void
}

; v2f16 = sitofp v2i64 is in sint_to_fp.i64.ll
