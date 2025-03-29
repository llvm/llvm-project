; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin < %s | FileCheck %s --check-prefix=MACHO

; CHECK-LABEL: .type _Z3funv,@function
; CHECK-NEXT:    .word   3238382334  // 0xc105cafe
; CHECK-NEXT:    .word   42
; CHECK-NEXT:  _Z3funv:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ret

; MACHO:      ltmp0:
; MACHO-NEXT:   .long 3238382334 ; 0xc105cafe
; MACHO-NEXT:   .long 42 ; 0x2a
; MACHO-NEXT:   .alt_entry __Z3funv
; MACHO-NEXT:   __Z3funv:
; MACHO-NEXT:   ; %bb.0:
; MACHO-NEXT:   ret

define dso_local void @_Z3funv() nounwind !func_sanitize !0 {
  ret void
}

!0 = !{i32 3238382334, i32 42}
