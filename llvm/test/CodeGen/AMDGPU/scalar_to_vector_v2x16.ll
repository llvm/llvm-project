; RUN: llc -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=fiji -O0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}scalar_to_vector_i16:
; GCN:   v_mov_b32_e32 [[V:v[0-9]+]], 42
; GCN: buffer_store_short [[V]],
define void @scalar_to_vector_i16() {
  %tmp = load <2 x i16>, ptr addrspace(5) undef
  %tmp1 = insertelement <2 x i16> %tmp, i16 42, i64 0
  store <2 x i16> %tmp1, ptr addrspace(5) undef
  ret void
}

; GCN-LABEL: {{^}}scalar_to_vector_f16:
; GCN:   v_mov_b32_e32 [[V:v[0-9]+]], 0x3c00
; GCN: buffer_store_short [[V]],
define void @scalar_to_vector_f16() {
  %tmp = load <2 x half>, ptr addrspace(5) undef
  %tmp1 = insertelement <2 x half> %tmp, half 1.0, i64 0
  store <2 x half> %tmp1, ptr addrspace(5) undef
  ret void
}
