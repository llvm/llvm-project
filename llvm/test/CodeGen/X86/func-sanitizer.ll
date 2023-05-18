; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; CHECK:      .type _Z3funv,@function
; CHECK-NEXT:   .long   3238382334  # 0xc105cafe
; CHECK-NEXT:   .long   42
; CHECK-NEXT: _Z3funv:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT:   # %bb.0:
; CHECK-NEXT:   retq

define dso_local void @_Z3funv() !func_sanitize !0 {
  ret void
}

!0 = !{i32 3238382334, i32 42}
