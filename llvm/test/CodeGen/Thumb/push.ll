; RUN: llc < %s -mtriple=thumb-apple-darwin -frame-pointer=all | FileCheck %s
; rdar://7268481

define void @t() nounwind {
; CHECK-LABEL: t:
; CHECK: push {r7, lr}
entry:
  call void asm sideeffect alignstack ".long 0xe7ffdefe", ""() nounwind
  ret void
}
