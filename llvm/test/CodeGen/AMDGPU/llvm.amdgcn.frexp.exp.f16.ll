; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare i16 @llvm.amdgcn.frexp.exp.i16.f16(half %a)

; GCN-LABEL: {{^}}frexp_exp_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; VI:  v_frexp_exp_i16_f16_e32 v[[R_I16:[0-9]+]], v[[A_F16]]
; GCN: buffer_store_short v[[R_I16]]
define amdgpu_kernel void @frexp_exp_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = call i16 @llvm.amdgcn.frexp.exp.i16.f16(half %a.val)
  store i16 %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}frexp_exp_f16_sext
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; VI:  v_frexp_exp_i16_f16_e32 v[[R_I16:[0-9]+]], v[[A_F16]]
; VI:  v_bfe_i32 v[[R_I32:[0-9]+]], v[[R_I16]], 0, 16{{$}}
; GCN: buffer_store_dword v[[R_I32]]
define amdgpu_kernel void @frexp_exp_f16_sext(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = call i16 @llvm.amdgcn.frexp.exp.i16.f16(half %a.val)
  %r.val.sext = sext i16 %r.val to i32
  store i32 %r.val.sext, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}frexp_exp_f16_zext
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; VI:  v_frexp_exp_i16_f16_e32 v[[R_I16:[0-9]+]], v[[A_F16]]
; GCN: buffer_store_dword v[[R_I16]]
define amdgpu_kernel void @frexp_exp_f16_zext(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = call i16 @llvm.amdgcn.frexp.exp.i16.f16(half %a.val)
  %r.val.zext = zext i16 %r.val to i32
  store i32 %r.val.zext, ptr addrspace(1) %r
  ret void
}
