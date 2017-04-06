; RUN: llc -march=amdgcn -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}fptoui_f16_to_i16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_u32_f32_e32 v[[R_I16:[0-9]+]], v[[A_F32]]
; VI:  v_cvt_i32_f32_e32 v[[R_I16:[0-9]+]], v[[A_F32]]
; GCN: buffer_store_short v[[R_I16]]
; GCN: s_endpgm
define amdgpu_kernel void @fptoui_f16_to_i16(
    i16 addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = fptoui half %a.val to i16
  store i16 %r.val, i16 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptoui_f16_to_i32
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; GCN: v_cvt_u32_f32_e32 v[[R_I32:[0-9]+]], v[[A_F32]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @fptoui_f16_to_i32(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = fptoui half %a.val to i32
  store i32 %r.val, i32 addrspace(1)* %r
  ret void
}

; Need to make sure we promote f16 to f32 when converting f16 to i64. Existing
; test checks code generated for 'i64 = fp_to_uint f32'.

; GCN-LABEL: {{^}}fptoui_f16_to_i64
; GCN: buffer_load_ushort
; GCN: v_cvt_f32_f16_e32
; GCN: s_endpgm
define amdgpu_kernel void @fptoui_f16_to_i64(
    i64 addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = fptoui half %a.val to i64
  store i64 %r.val, i64 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptoui_v2f16_to_v2i16
; GCN:     buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN:     v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; GCN-DAG: v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; GCN-DAG: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI:      v_cvt_u32_f32_e32 v[[R_I16_1:[0-9]+]], v[[A_F32_1]]
; SI:      v_cvt_u32_f32_e32 v[[R_I16_0:[0-9]+]], v[[A_F32_0]]
; VI:      v_cvt_i32_f32_e32 v[[R_I16_0:[0-9]+]], v[[A_F32_0]]
; VI:      v_cvt_i32_f32_e32 v[[R_I16_1:[0-9]+]], v[[A_F32_1]]
; GCN:     v_lshlrev_b32_e32 v[[R_I16_HI:[0-9]+]], 16, v[[R_I16_1]]
; GCN:     v_or_b32_e32 v[[R_V2_I16:[0-9]+]], v[[R_I16_HI]], v[[R_I16_0]]
; GCN:     buffer_store_dword v[[R_V2_I16]]
; GCN:     s_endpgm
define amdgpu_kernel void @fptoui_v2f16_to_v2i16(
    <2 x i16> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %r.val = fptoui <2 x half> %a.val to <2 x i16>
  store <2 x i16> %r.val, <2 x i16> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptoui_v2f16_to_v2i32
; GCN: buffer_load_dword
; GCN: v_cvt_f32_f16_e32
; GCN: v_cvt_f32_f16_e32
; GCN: v_cvt_u32_f32_e32
; GCN: v_cvt_u32_f32_e32
; GCN: buffer_store_dwordx2
; GCN: s_endpgm
define amdgpu_kernel void @fptoui_v2f16_to_v2i32(
    <2 x i32> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %r.val = fptoui <2 x half> %a.val to <2 x i32>
  store <2 x i32> %r.val, <2 x i32> addrspace(1)* %r
  ret void
}

; Need to make sure we promote f16 to f32 when converting f16 to i64. Existing
; test checks code generated for 'i64 = fp_to_uint f32'.

; GCN-LABEL: {{^}}fptoui_v2f16_to_v2i64
; GCN: buffer_load_dword
; GCN: v_cvt_f32_f16_e32
; GCN: v_cvt_f32_f16_e32
; GCN: s_endpgm
define amdgpu_kernel void @fptoui_v2f16_to_v2i64(
    <2 x i64> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %r.val = fptoui <2 x half> %a.val to <2 x i64>
  store <2 x i64> %r.val, <2 x i64> addrspace(1)* %r
  ret void
}
