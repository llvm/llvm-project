; RUN: llc -verify-machineinstrs -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefixes=GCN,SI -check-prefix=FUNC %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefixes=GCN,VI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}selectcc_i64:
; EG: XOR_INT
; EG: XOR_INT
; EG: OR_INT
; EG: CNDE_INT
; EG: CNDE_INT
; SI: v_cmp_eq_u64_e32
; VI: s_cmp_eq_u64
; VI: s_cselect_b64 vcc, 1, 0
; GCN: v_cndmask
; GCN: v_cndmask
define amdgpu_kernel void @selectcc_i64(i64 addrspace(1) * %out, i64 %lhs, i64 %rhs, i64 %true, i64 %false) {
entry:
  %0 = icmp eq i64 %lhs, %rhs
  %1 = select i1 %0, i64 %true, i64 %false
  store i64 %1, i64 addrspace(1)* %out
  ret void
}
