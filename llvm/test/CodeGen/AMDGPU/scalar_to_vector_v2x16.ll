; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-OPT %s
; RUN: llc -march=amdgcn -mcpu=fiji -O0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-NOOPT %s

; GCN-LABEL: {{^}}scalar_to_vector_i16:
; GCN-NOOPT: s_mov_b32 [[S:s[0-9]+]], 42
; GCN-NOOPT: v_mov_b32_e32 [[V:v[0-9]+]], [[S]]
; GCN-OPT:   v_mov_b32_e32 [[V:v[0-9]+]], 42
; GCN: buffer_store_short [[V]],
define void @scalar_to_vector_i16() {
  %tmp = load <2 x i16>, <2 x i16> addrspace(5)* undef
  %tmp1 = insertelement <2 x i16> %tmp, i16 42, i64 0
  store <2 x i16> %tmp1, <2 x i16> addrspace(5)* undef
  ret void
}

; GCN-LABEL: {{^}}scalar_to_vector_f16:
; GCN-NOOPT: s_mov_b32 [[S:s[0-9]+]], 0x3c00
; GCN-NOOPT: v_mov_b32_e32 [[V:v[0-9]+]], [[S]]
; GCN-OPT:   v_mov_b32_e32 [[V:v[0-9]+]], 0x3c00
; GCN: buffer_store_short [[V]],
define void @scalar_to_vector_f16() {
  %tmp = load <2 x half>, <2 x half> addrspace(5)* undef
  %tmp1 = insertelement <2 x half> %tmp, half 1.0, i64 0
  store <2 x half> %tmp1, <2 x half> addrspace(5)* undef
  ret void
}
