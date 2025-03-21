; RUN: llc -mtriple=x86_64-linux-gnu %s -o - 2>&1 | FileCheck %s

declare void @callee()
define void @caller() sspreq {
; CHECK: callq   callee@PLT
; CHECK: callq   callee@PLT
; CHECK: cmpq
; CHECK: jne
; CHECK: callq   __stack_chk_fail@PLT

  tail call void @callee()
  call void @callee()
  ret void
}
