; RUN: llc -mtriple=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare half @llvm.amdgcn.sin.f16(half %a)

; GCN-LABEL: {{^}}sin_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; VI:  v_sin_f16_e32 v[[R_F16:[0-9]+]], v[[A_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @sin_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = call half @llvm.amdgcn.sin.f16(half %a.val)
  store half %r.val, ptr addrspace(1) %r
  ret void
}
