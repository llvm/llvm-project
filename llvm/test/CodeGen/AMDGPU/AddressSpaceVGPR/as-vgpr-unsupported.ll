; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s

; Accesses of the VGPR "as memory" address space (13) that are not implemented
; must be rejected with a clean diagnostic on both SelectionDAG and GlobalISel,
; rather than failing with "cannot select" / "unable to legalize" - or, worse,
; silently generating wrong code. Dword-aligned whole-dword and naturally
; aligned 8-/16-bit accesses are implemented; see as-vgpr-basic.ll and
; as-vgpr-bits.ll.

; A whole-dword access addresses registers by the dword index pointer >> 2,
; which discards the low two bits rather than accounting for them. An
; under-aligned one would therefore access the dword containing the address
; instead of the bytes asked for, reading or writing the wrong data with
; nothing to show for it, so it has to be rejected here.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and naturally aligned 8-/16-bit loads and stores are implemented
define i32 @load_i32_align1(ptr addrspace(13) inreg %p) {
  %x = load i32, ptr addrspace(13) %p, align 1
  ret i32 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and naturally aligned 8-/16-bit loads and stores are implemented
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

; An under-aligned 16-bit access may straddle a dword boundary, which the
; bit-field extract / insert lowering cannot express.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and naturally aligned 8-/16-bit loads and stores are implemented
define i32 @load_i16_align1(ptr addrspace(13) inreg %p) {
  %x = load i16, ptr addrspace(13) %p, align 1
  %y = zext i16 %x to i32
  ret i32 %y
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and naturally aligned 8-/16-bit loads and stores are implemented
define void @store_i16_align1(ptr addrspace(13) inreg %p, i16 %v) {
  store i16 %v, ptr addrspace(13) %p, align 1
  ret void
}

; A sub-dword load extended into a value wider than a dword.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and naturally aligned 8-/16-bit loads and stores are implemented
define i64 @load_i8_zext_i64(ptr addrspace(13) inreg %p) {
  %x = load i8, ptr addrspace(13) %p
  %y = zext i8 %x to i64
  ret i64 %y
}

; A memory size that is neither a whole dword nor 8/16 bits.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and naturally aligned 8-/16-bit loads and stores are implemented
define i1 @load_i1(ptr addrspace(13) inreg %p) {
  %x = load i1, ptr addrspace(13) %p
  ret i1 %x
}

; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and naturally aligned 8-/16-bit loads and stores are implemented
define void @store_i1(ptr addrspace(13) inreg %p, i1 %v) {
  store i1 %v, ptr addrspace(13) %p
  ret void
}

; A whole-dword size with no corresponding V_LOAD_IDX/V_STORE_IDX pseudo.
; CHECK: error: {{.*}}unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword and naturally aligned 8-/16-bit loads and stores are implemented
define <14 x i32> @load_v14i32(ptr addrspace(13) inreg %p) {
  %x = load <14 x i32>, ptr addrspace(13) %p
  ret <14 x i32> %x
}

; And an 8-bit access is held only to its own natural alignment, which is one
; byte - the whole-dword rule must not catch it.
; CHECK-NOT: in function load_i8_align1
define i8 @load_i8_align1(ptr addrspace(13) inreg %p) {
  %x = load i8, ptr addrspace(13) %p, align 1
  ret i8 %x
}
