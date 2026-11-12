; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s --check-prefix=MACHO

; CHECK:      .type _Z3funv,@function
; CHECK-NEXT:   .long   3238382334  # 0xc105cafe
; CHECK-NEXT:   .long   42
; CHECK-NEXT: _Z3funv:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT:   # %bb.0:
; CHECK-NEXT:   retq

; MACHO:      ltmp0:
; MACHO-NEXT:  .long 3238382334 ## 0xc105cafe
; MACHO-NEXT:  .long 42 ## 0x2a
; MACHO-NEXT:  .alt_entry __Z3funv
; MACHO-NEXT: __Z3funv:
; MACHO-NEXT:  .cfi_startproc
; MACHO-NEXT:  # %bb.0:
; MACHO-NEXT:  retq

define dso_local void @_Z3funv() !func_sanitize !0 {
  ret void
}

!0 = !{i32 3238382334, i32 42}
