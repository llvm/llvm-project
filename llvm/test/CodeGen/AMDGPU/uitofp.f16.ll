; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}uitofp_i16_to_f16
; GCN: buffer_load_ushort v[[A_I16:[0-9]+]]
; SI:  v_cvt_f32_u32_e32 v[[A_F32:[0-9]+]], v[[A_I16]]
; SI: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_F32]]

; VI:  v_cvt_f16_u16_e32 v[[R_F16:[0-9]+]], v[[A_I16]]

; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @uitofp_i16_to_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load i16, ptr addrspace(1) %a
  %r.val = uitofp i16 %a.val to half
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}uitofp_i32_to_f16
; GCN: buffer_load_dword v[[A_I32:[0-9]+]]
; GCN: v_cvt_f32_u32_e32 v[[A_I16:[0-9]+]], v[[A_I32]]
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_I16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @uitofp_i32_to_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load i32, ptr addrspace(1) %a
  %r.val = uitofp i32 %a.val to half
  store half %r.val, ptr addrspace(1) %r
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


; VI-DAG: v_cvt_f16_u16_e32
; VI-DAG: v_cvt_f16_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1
; VI:     v_or_b32_e32

; GCN: buffer_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @uitofp_v2i16_to_v2f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load <2 x i16>, ptr addrspace(1) %a
  %r.val = uitofp <2 x i16> %a.val to <2 x half>
  store <2 x half> %r.val, ptr addrspace(1) %r
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
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load <2 x i32>, ptr addrspace(1) %a
  %r.val = uitofp <2 x i32> %a.val to <2 x half>
  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}s_uint_to_fp_i1_to_f16:
; GCN-DAG: v_cmp_le_f32_e32 [[CMP0:vcc]], 1.0, {{v[0-9]+}}
; GCN-DAG: v_cmp_le_f32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], 0, {{v[0-9]+}}
; GCN: s_xor_b64 [[R_CMP:s\[[0-9]+:[0-9]+\]]], [[CMP1]], [[CMP0]]
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1.0, [[R_CMP]]
; GCN-NEXT: v_cvt_f16_f32_e32 [[R_F16:v[0-9]+]], [[RESULT]]
; GCN: buffer_store_short
; GCN: s_endpgm
define amdgpu_kernel void @s_uint_to_fp_i1_to_f16(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load float, ptr addrspace(1) %in0
  %b = load float, ptr addrspace(1) %in1
  %acmp = fcmp oge float %a, 0.000000e+00
  %bcmp = fcmp oge float %b, 1.000000e+00
  %result = xor i1 %acmp, %bcmp
  %fp = uitofp i1 %result to half
  store half %fp, ptr addrspace(1) %out
  ret void
}

; f16 = uitofp i64 is in uint_to_fp.i64.ll
