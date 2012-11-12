; RUN: llc < %s -mtriple=thumb-apple-darwin -disable-fp-elim | FileCheck %s
; rdar://7268481

define void @t() nounwind {
; CHECK: t:
; CHECK: push {r7}
entry:
  call void asm sideeffect alignstack ".long 0xe7ffdefe", ""() nounwind
  ret void
}
