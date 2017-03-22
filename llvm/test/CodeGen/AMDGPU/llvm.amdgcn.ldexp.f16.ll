; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare half @llvm.amdgcn.ldexp.f16(half %a, i32 %b)

; GCN-LABEL: {{^}}ldexp_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_I32:[0-9]+]]
; VI: v_ldexp_f16_e32 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_I32]]
; GCN: buffer_store_short v[[R_F16]]
define amdgpu_kernel void @ldexp_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    i32 addrspace(1)* %b) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load i32, i32 addrspace(1)* %b
  %r.val = call half @llvm.amdgcn.ldexp.f16(half %a.val, i32 %b.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}ldexp_f16_imm_a
; GCN: buffer_load_dword v[[B_I32:[0-9]+]]
; VI: v_ldexp_f16_e32 v[[R_F16:[0-9]+]], 2.0, v[[B_I32]]
; GCN: buffer_store_short v[[R_F16]]
define amdgpu_kernel void @ldexp_f16_imm_a(
    half addrspace(1)* %r,
    i32 addrspace(1)* %b) {
  %b.val = load i32, i32 addrspace(1)* %b
  %r.val = call half @llvm.amdgcn.ldexp.f16(half 2.0, i32 %b.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}ldexp_f16_imm_b
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; VI: v_ldexp_f16_e64 v[[R_F16:[0-9]+]], v[[A_F16]], 2{{$}}
; GCN: buffer_store_short v[[R_F16]]
define amdgpu_kernel void @ldexp_f16_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a) {
  %a.val = load half, half addrspace(1)* %a
  %r.val = call half @llvm.amdgcn.ldexp.f16(half %a.val, i32 2)
  store half %r.val, half addrspace(1)* %r
  ret void
}
