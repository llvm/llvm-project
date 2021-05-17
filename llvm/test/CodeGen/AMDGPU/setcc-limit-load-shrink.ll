; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}const_load_no_shrink_dword_to_unaligned_byte:
; GCN: s_load_dword s{{[0-9]+}}
; GCN: s_load_dword [[LD:s[0-9]+]],
; GCN: s_bfe_i32 s{{[0-9]+}}, [[LD]], 0x10013
define amdgpu_kernel void @const_load_no_shrink_dword_to_unaligned_byte(i32 addrspace(1)* %out, i32 addrspace(4)* %in, i32 %x) {
  %ptr = getelementptr i32, i32 addrspace(4)* %in, i32 %x
  %load = load i32, i32 addrspace(4)* %ptr, align 4
  %and = and i32 %load, 524288
  %cmp = icmp eq i32 %and, 0
  %sel = select i1 %cmp, i32 0, i32 -1
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: const_load_no_shrink_dword_to_aligned_byte:
; GCN: s_load_dword s{{[0-9]+}}
; GCN: s_load_dword [[LD:s[0-9]+]],
; GCN: s_bfe_i32 s{{[0-9]+}}, [[LD]], 0x10003
define amdgpu_kernel void @const_load_no_shrink_dword_to_aligned_byte(i32 addrspace(1)* %out, i32 addrspace(4)* %in, i32 %x) {
  %ptr = getelementptr i32, i32 addrspace(4)* %in, i32 %x
  %load = load i32, i32 addrspace(4)* %ptr, align 4
  %and = and i32 %load, 8
  %cmp = icmp eq i32 %and, 0
  %sel = select i1 %cmp, i32 0, i32 -1
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: global_load_no_shrink_dword_to_unaligned_byte:
; GCN: s_load_dword s{{[0-9]+}}
; GCN: s_load_dword [[LD:s[0-9]+]],
; GCN: s_bfe_i32 s{{[0-9]+}}, [[LD]], 0x10013
define amdgpu_kernel void @global_load_no_shrink_dword_to_unaligned_byte(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 %x) {
  %ptr = getelementptr i32, i32 addrspace(1)* %in, i32 %x
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %and = and i32 %load, 524288
  %cmp = icmp eq i32 %and, 0
  %sel = select i1 %cmp, i32 0, i32 -1
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: global_load_no_shrink_dword_to_aligned_byte:
; GCN: s_load_dword s{{[0-9]+}}
; GCN: s_load_dword [[LD:s[0-9]+]],
; GCN: s_bfe_i32 s{{[0-9]+}}, [[LD]], 0x10003
define amdgpu_kernel void @global_load_no_shrink_dword_to_aligned_byte(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %x) {
  %ptr = getelementptr i32, i32 addrspace(1)* %in, i32 %x
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %and = and i32 %load, 8
  %cmp = icmp eq i32 %and, 0
  %sel = select i1 %cmp, i32 0, i32 -1
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: const_load_shrink_dword_to_unaligned_byte:
; GCN: global_load_ushort
define amdgpu_kernel void @const_load_shrink_dword_to_unaligned_byte(i32 addrspace(1)* %out, i32 addrspace(4)* %in, i32 %x) {
  %ptr = getelementptr i32, i32 addrspace(4)* %in, i32 %x
  %load = load i32, i32 addrspace(4)* %ptr, align 2
  %and = and i32 %load, 524288
  %cmp = icmp eq i32 %and, 0
  %sel = select i1 %cmp, i32 0, i32 -1
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}
