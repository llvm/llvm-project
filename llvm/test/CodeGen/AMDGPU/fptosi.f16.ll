; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}fptosi_f16_to_i16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; SI: v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI: v_cvt_i32_f32_e32 v[[R_I16:[0-9]+]], v[[A_F32]]
; VI: v_cvt_i16_f16_e32 v[[R_I16:[0-9]+]], v[[A_F16]]
; GCN: buffer_store_short v[[R_I16]]
; GCN: s_endpgm
define amdgpu_kernel void @fptosi_f16_to_i16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = fptosi half %a.val to i16
  store i16 %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fptosi_f16_to_i32
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; GCN: v_cvt_i32_f32_e32 v[[R_I32:[0-9]+]], v[[A_F32]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fptosi_f16_to_i32(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = fptosi half %a.val to i32
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}

; Need to make sure we promote f16 to f32 when converting f16 to i64. Existing
; test checks code generated for 'i64 = fp_to_sint f32'.

; GCN-LABEL: {{^}}fptosi_f16_to_i64
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; GCN: v_cvt_i32_f32_e32 v[[R_I64_Low:[0-9]+]], v[[A_F32]]
; GCN: v_ashrrev_i32_e32 v[[R_I64_High:[0-9]+]], 31, v[[R_I64_Low]]
; GCN: buffer_store_dwordx2 v[[[R_I64_Low]]{{\:}}[[R_I64_High]]]
; GCN: s_endpgm
define amdgpu_kernel void @fptosi_f16_to_i64(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = fptosi half %a.val to i64
  store i64 %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fptosi_v2f16_to_v2i16
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]

; SI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI: v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI: v_cvt_i32_f32_e32 v[[R_I16_0:[0-9]+]], v[[A_F32_0]]
; SI-DAG: v_cvt_i32_f32_e32 v[[R_I16_1:[0-9]+]], v[[A_F32_1]]
; SI-DAG: v_and_b32_e32 v[[R_I16_LO:[0-9]+]], 0xffff, v[[R_I16_0]]
; SI: v_lshlrev_b32_e32 v[[R_I16_HI:[0-9]+]], 16, v[[R_I16_1]]
; SI: v_or_b32_e32 v[[R_V2_I16:[0-9]+]], v[[R_I16_LO]], v[[R_I16_HI]]

; VI: v_cvt_i16_f16_e32 v[[A_I16_0:[0-9]+]], v[[A_V2_F16]]
; VI: v_cvt_i16_f16_sdwa v[[A_I16_1:[0-9]+]], v[[A_V2_F16]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1
; VI: v_or_b32_sdwa v[[R_V2_I16:[0-9]+]], v[[A_I16_0]], v[[A_I16_1]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD

; GCN: buffer_store_dword v[[R_V2_I16]]
; GCN: s_endpgm

define amdgpu_kernel void @fptosi_v2f16_to_v2i16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %r.val = fptosi <2 x half> %a.val to <2 x i16>
  store <2 x i16> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fptosi_v2f16_to_v2i32
; GCN: buffer_load_dword
; GCN: v_cvt_f32_f16_e32
; SI: v_cvt_f32_f16_e32
; VI: v_cvt_f32_f16_sdwa
; GCN: v_cvt_i32_f32_e32
; GCN: v_cvt_i32_f32_e32
; GCN: buffer_store_dwordx2
; GCN: s_endpgm
define amdgpu_kernel void @fptosi_v2f16_to_v2i32(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %r.val = fptosi <2 x half> %a.val to <2 x i32>
  store <2 x i32> %r.val, ptr addrspace(1) %r
  ret void
}

; Need to make sure we promote f16 to f32 when converting f16 to i64. Existing
; test checks code generated for 'i64 = fp_to_sint f32'.

; GCN-LABEL: {{^}}fptosi_v2f16_to_v2i64
; GCN: buffer_load_dword v[[A_F16_0:[0-9]+]]
; SI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_F16_0]]
; SI: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_F16_0]]
; SI: v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI: v_cvt_i32_f32_e32 v[[R_I64_0_Low:[0-9]+]], v[[A_F32_0]]
; SI: v_ashrrev_i32_e32 v[[R_I64_0_High:[0-9]+]], 31, v[[R_I64_0_Low]]
; SI: v_cvt_i32_f32_e32 v[[R_I64_1_Low:[0-9]+]], v[[A_F32_1]]
; SI: v_ashrrev_i32_e32 v[[R_I64_1_High:[0-9]+]], 31, v[[R_I64_1_Low]]
; VI-DAG: v_cvt_f32_f16_sdwa v[[A_F32_1:[0-9]+]], v[[A_F16_0]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
; VI-DAG: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_F16_0]]
; VI-DAG: v_cvt_i32_f32_e32 v[[R_I64_1_Low:[0-9]+]], v[[A_F32_1]]
; VI-DAG: v_cvt_i32_f32_e32 v[[R_I64_0_Low:[0-9]+]], v[[A_F32_0]]
; VI-NOT: DEADBEEF
; VI-DAG: v_ashrrev_i32_e32 v[[R_I64_1_High:[0-9]+]], 31, v[[R_I64_1_Low]]
; VI-DAG: v_ashrrev_i32_e32 v[[R_I64_0_High:[0-9]+]], 31, v[[R_I64_0_Low]]
; GCN: buffer_store_dwordx4 v[[[R_I64_0_Low]]{{\:}}[[R_I64_1_High]]]
; GCN: s_endpgm
define amdgpu_kernel void @fptosi_v2f16_to_v2i64(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %r.val = fptosi <2 x half> %a.val to <2 x i64>
  store <2 x i64> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fptosi_f16_to_i1:
; SI: v_cvt_f32_f16_e32 v{{[0-9]+}}, s{{[0-9]+}}
; SI: v_cmp_eq_f32_e32 vcc, -1.0, v{{[0-9]+}}
; SI: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
; VI: v_cmp_eq_f16_e64 s{{\[[0-9]+:[0-9]+\]}}, 0xbc00, s{{[0-9]+}}
; VI: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, s[4:5]
define amdgpu_kernel void @fptosi_f16_to_i1(ptr addrspace(1) %out, half %in) {
entry:
  %conv = fptosi half %in to i1
  store i1 %conv, ptr addrspace(1) %out
  ret void
}
