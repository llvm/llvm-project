; RUN: not --crash llc < %s -mtriple=aarch64-windows -o /dev/null 2>&1 | FileCheck %s

; Authenticating the return address needs SP to be the same as when it was
; signed, which isn't the case when the tail call changes the stack argument
; size, and Windows unwind info can't describe the sequence that works around it.

; CHECK: Can't handle a tail call that changes the stack argument size in a function that signs its return address on Windows

declare swifttailcc void @callee12(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare void @clobber()

define swifttailcc void @pac_grow(i64 %a) "sign-return-address"="all" {
  call void @clobber()
  musttail call swifttailcc void @callee12(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  ret void
}
