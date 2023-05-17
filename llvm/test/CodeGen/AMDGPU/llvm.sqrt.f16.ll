; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare half @llvm.sqrt.f16(half %a)
declare <2 x half> @llvm.sqrt.v2f16(<2 x half> %a)

; GCN-LABEL: {{^}}sqrt_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_sqrt_f32_e32 v[[R_F32:[0-9]+]], v[[A_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_sqrt_f16_e32 v[[R_F16:[0-9]+]], v[[A_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @sqrt_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = call half @llvm.sqrt.f16(half %a.val)
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}sqrt_v2f16
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; SI:  v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI:  v_sqrt_f32_e32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]]
; SI:  v_sqrt_f32_e32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI-NOT: v_and_b32
; SI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; VI-DAG: v_sqrt_f16_e32 v[[R_F16_0:[0-9]+]], v[[A_V2_F16]]
; VI-DAG: v_sqrt_f16_sdwa v[[R_F16_1:[0-9]+]], v[[A_V2_F16]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1
; VI-NOT: v_and_b32
; VI:     v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_1]]

; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @sqrt_v2f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %r.val = call <2 x half> @llvm.sqrt.v2f16(<2 x half> %a.val)
  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}
