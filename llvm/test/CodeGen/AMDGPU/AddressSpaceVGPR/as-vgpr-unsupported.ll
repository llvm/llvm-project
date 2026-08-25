; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s

; Accesses of the VGPR "as memory" address space (13) that are not implemented
; must be rejected with a clean diagnostic on both SelectionDAG and GlobalISel,
; rather than failing with "cannot select" / "unable to legalize" - or, worse,
; silently generating wrong code.

; Sub-dword (8/16-bit) accesses are not yet implemented; support lands in a
; later change.
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

; An access addresses registers by the dword index pointer >> 2, which discards
; the low two bits rather than accounting for them. An under-aligned one would
; therefore reach the dword containing the address instead of the bytes asked
; for - the same code as a correctly aligned access, reading the wrong data with
; nothing to show for it.
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

; Alignment is required of the pointer, not of the accessed type: a 64-bit
; access needs only the dword alignment the index computation relies on.
; CHECK-NOT: in function load_i64_align4
define i64 @load_i64_align4(ptr addrspace(13) inreg %p) {
  %x = load i64, ptr addrspace(13) %p, align 4
  ret i64 %x
}
