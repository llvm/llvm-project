; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: .type _Z3funv,@function
; CHECK-NEXT:    .word   3238382334  // 0xc105cafe
; CHECK-NEXT:    .word   42
; CHECK-NEXT:  _Z3funv:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ret

define dso_local void @_Z3funv() nounwind !func_sanitize !0 {
  ret void
}

!0 = !{i32 3238382334, i32 42}
