; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare i32 @llvm.amdgcn.frexp.exp.f16(half %a)

; GCN-LABEL: {{^}}frexp_exp_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; VI: v_frexp_exp_i16_f16_e32 v[[R_I16:[0-9]+]], v[[A_F16]]
; GCN: buffer_store_short v[[R_I16]]
define void @frexp_exp_f16(
    i16 addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = call i32 @llvm.amdgcn.frexp.exp.f16(half %a.val)
  %r.val.i16 = trunc i32 %r.val to i16
  store i16 %r.val.i16, i16 addrspace(1)* %r
  ret void
}
