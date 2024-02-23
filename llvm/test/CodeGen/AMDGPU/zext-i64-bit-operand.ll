; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}zext_or_operand_i64:
; GCN: buffer_load_dwordx2 v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GCN: buffer_load_dword v[[LD32:[0-9]+]]
; GCN-NOT: _or_
; GCN-NOT: v[[HI]]
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: v_or_b32_e32 v[[LO]], v[[LO]], v[[LD32]]
; GCN-NOT: _or_
; GCN-NOT: v[[HI]]
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @zext_or_operand_i64(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %ld.64 = load volatile i64, ptr addrspace(1) %in0
  %ld.32 = load volatile i32, ptr addrspace(1) %in1
  %ext = zext i32 %ld.32 to i64
  %or = or i64 %ld.64, %ext
  store i64 %or, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}zext_or_operand_commute_i64:
; GCN: buffer_load_dwordx2 v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GCN: buffer_load_dword v[[LD32:[0-9]+]]
; GCN-NOT: _or_
; GCN-NOT: v[[HI]]
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: v_or_b32_e32 v[[LO]], v[[LO]], v[[LD32]]
; GCN-NOT: v[[HI]]
; GCN-NOT: _or_
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @zext_or_operand_commute_i64(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %ld.64 = load volatile i64, ptr addrspace(1) %in0
  %ld.32 = load volatile i32, ptr addrspace(1) %in1
  %ext = zext i32 %ld.32 to i64
  %or = or i64 %ext, %ld.64
  store i64 %or, ptr addrspace(1) %out
  ret void
}
