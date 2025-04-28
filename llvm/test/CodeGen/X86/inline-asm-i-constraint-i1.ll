; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Make sure that boolean immediates are properly (zero) extended.
; CHECK: .Ltmp[[N:[0-9]+]]:
; CHECK-NEXT: .quad (42+1)-.Ltmp[[N]]

target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() #0 {
entry:
  tail call void asm sideeffect ".quad 42 + ${0:c} - .\0A\09", "i,~{dirflag},~{fpsr},~{flags}"(i1 true) #0
  ret i32 1
}

attributes #0 = { nounwind }
