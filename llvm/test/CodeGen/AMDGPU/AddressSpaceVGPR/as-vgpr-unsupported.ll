; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s

; Accesses of the VGPR "as memory" address space (13) that are still not
; implemented must be rejected with a clean diagnostic on both SelectionDAG and
; GlobalISel, rather than failing with "cannot select" / "unable to legalize".
; Dword-aligned whole-dword and 8-/16-bit accesses are implemented; see
; as-vgpr-basic.ll and
; as-vgpr-bits.ll.

; A sub-dword load extended into a value wider than a dword.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and 8-/16-bit loads and stores are implemented
define i64 @load_i8_zext_i64(ptr addrspace(13) inreg %p) {
  %x = load i8, ptr addrspace(13) %p
  %y = zext i8 %x to i64
  ret i64 %y
}

; A memory size that is neither a whole dword nor 8/16 bits.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and 8-/16-bit loads and stores are implemented
define i1 @load_i1(ptr addrspace(13) inreg %p) {
  %x = load i1, ptr addrspace(13) %p
  ret i1 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and 8-/16-bit loads and stores are implemented
define void @store_i1(ptr addrspace(13) inreg %p, i1 %v) {
  store i1 %v, ptr addrspace(13) %p
  ret void
}

; A whole-dword size with no corresponding V_LOAD_IDX/V_STORE_IDX pseudo.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and 8-/16-bit loads and stores are implemented
define <14 x i32> @load_v14i32(ptr addrspace(13) inreg %p) {
  %x = load <14 x i32>, ptr addrspace(13) %p
  ret <14 x i32> %x
}

; A whole-dword access addresses registers by the dword index pointer >> 2,
; which discards the low two bits rather than accounting for them. An
; under-aligned one would therefore reach the dword containing the address
; instead of the bytes asked for - the same code as a correctly aligned access,
; reading the wrong data with nothing to show for it. A sub-dword access
; computes a bit offset instead, and is held only to its own natural alignment.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and 8-/16-bit loads and stores are implemented
define i32 @load_i32_align1(ptr addrspace(13) inreg %p) {
  %x = load i32, ptr addrspace(13) %p, align 1
  ret i32 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and 8-/16-bit loads and stores are implemented
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

; And a byte access at align 1 is fine, which is what the whole-dword-only
; restriction must not catch.
; CHECK-NOT: in function load_i8_align1
define i8 @load_i8_align1(ptr addrspace(13) inreg %p) {
  %x = load i8, ptr addrspace(13) %p, align 1
  ret i8 %x
}
