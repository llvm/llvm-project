; RUN: llc -march=amdgcn -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}fpext_f16_to_f32
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: v_cvt_f32_f16_e32 v[[R_F32:[0-9]+]], v[[A_F16]]
; GCN: buffer_store_dword v[[R_F32]]
; GCN: s_endpgm
define void @fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = fpext half %a.val to float
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fpext_f16_to_f64
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; GCN: v_cvt_f64_f32_e32 v{{\[}}[[R_F64_0:[0-9]+]]:[[R_F64_1:[0-9]+]]{{\]}}, v[[A_F32]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_F64_0]]:[[R_F64_1]]{{\]}}
; GCN: s_endpgm
define void @fpext_f16_to_f64(
    double addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = fpext half %a.val to double
  store double %r.val, double addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fpext_v2f16_to_v2f32
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; VI:  v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; GCN: v_cvt_f32_f16_e32 v[[R_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI:  v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; GCN: v_cvt_f32_f16_e32 v[[R_F32_1:[0-9]+]], v[[A_F16_1]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_F32_0]]:[[R_F32_1]]{{\]}}
; GCN: s_endpgm
define void @fpext_v2f16_to_v2f32(
    <2 x float> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %r.val = fpext <2 x half> %a.val to <2 x float>
  store <2 x float> %r.val, <2 x float> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fpext_v2f16_to_v2f64
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; GCN: v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; GCN: v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; GCN: v_cvt_f64_f32_e32 v{{\[}}{{[0-9]+}}:[[R_F64_3:[0-9]+]]{{\]}}, v[[A_F32_1]]
; GCN: v_cvt_f64_f32_e32 v{{\[}}[[R_F64_0:[0-9]+]]:{{[0-9]+}}{{\]}}, v[[A_F32_0]]
; GCN: buffer_store_dwordx4 v{{\[}}[[R_F64_0]]:[[R_F64_3]]{{\]}}
; GCN: s_endpgm
define void @fpext_v2f16_to_v2f64(
    <2 x double> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %r.val = fpext <2 x half> %a.val to <2 x double>
  store <2 x double> %r.val, <2 x double> addrspace(1)* %r
  ret void
}
