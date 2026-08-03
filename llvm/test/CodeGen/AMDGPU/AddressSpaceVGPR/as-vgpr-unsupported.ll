; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s

; Sub-dword (8/16-bit) accesses of the VGPR "as memory" address space (13) are
; not yet implemented. They must be rejected with a clean diagnostic on both
; SelectionDAG and GlobalISel, rather than failing with "cannot select" /
; "unable to legalize".

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only whole-dword loads and stores are implemented
define i8 @load_i8(ptr addrspace(13) inreg %p) {
  %x = load i8, ptr addrspace(13) %p
  ret i8 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only whole-dword loads and stores are implemented
define i16 @load_i16(ptr addrspace(13) inreg %p) {
  %x = load i16, ptr addrspace(13) %p
  ret i16 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only whole-dword loads and stores are implemented
define void @store_i8(ptr addrspace(13) inreg %p, i8 %v) {
  store i8 %v, ptr addrspace(13) %p
  ret void
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only whole-dword loads and stores are implemented
define void @store_i16(ptr addrspace(13) inreg %p, i16 %v) {
  store i16 %v, ptr addrspace(13) %p
  ret void
}
