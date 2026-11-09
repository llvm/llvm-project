; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s

; The exit thunk for a function taking and returning a 64-byte vector
; realigns the stack. X9 holds the x64 callee for
; __os_arm64x_dispatch_call_no_redirect, so it must not be used as the
; realignment scratch register.

declare <8 x i64> @ext(<8 x i64>)

define <8 x i64> @wrap(<8 x i64> %a) {
; CHECK-LABEL: $iexit_thunk$cdecl$m64$m64:
; CHECK:         .seh_endprologue
; CHECK-NEXT:    sub x0, sp, #240
; CHECK-NEXT:    and sp, x0, #0xffffffffffffffc0
; CHECK-NOT:     x9
; CHECK:         blr x16
  %r = call <8 x i64> @ext(<8 x i64> %a)
  ret <8 x i64> %r
}
