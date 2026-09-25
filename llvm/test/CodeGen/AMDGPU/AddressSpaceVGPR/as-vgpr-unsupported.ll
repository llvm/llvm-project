; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null < %s 2>&1 | FileCheck %s

; Unimplemented accesses get a clean diagnostic on both selectors, not a
; selection or legalization failure, or wrong code.

; Sub-dword accesses are not implemented yet.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i8 @load_i8(ptr addrspace(13) inreg %p) {
  %x = load i8, ptr addrspace(13) %p
  ret i8 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i16 @load_i16(ptr addrspace(13) inreg %p) {
  %x = load i16, ptr addrspace(13) %p
  ret i16 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define void @store_i8(ptr addrspace(13) inreg %p, i8 %v) {
  store i8 %v, ptr addrspace(13) %p
  ret void
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define void @store_i16(ptr addrspace(13) inreg %p, i16 %v) {
  store i16 %v, ptr addrspace(13) %p
  ret void
}

; The index is the pointer >> 2, so an under-aligned access would silently reach
; the containing dword.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i32 @load_i32_align1(ptr addrspace(13) inreg %p) {
  %x = load i32, ptr addrspace(13) %p, align 1
  ret i32 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define void @store_i32_align1(ptr addrspace(13) inreg %p, i32 %v) {
  store i32 %v, ptr addrspace(13) %p, align 1
  ret void
}

; Only dword alignment is required, whatever the access size.
; CHECK-NOT: in function load_i64_align4
define i64 @load_i64_align4(ptr addrspace(13) inreg %p) {
  %x = load i64, ptr addrspace(13) %p, align 4
  ret i64 %x
}
