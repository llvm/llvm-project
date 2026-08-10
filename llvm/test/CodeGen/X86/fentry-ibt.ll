; RUN: llc %s -o - -verify-machineinstrs -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: @test1
; CHECK: endbr64
; CHECK: callq __fentry__
; CHECK-NOT: mcount
; CHECK: retq
}

!llvm.module.flags = !{!0}

attributes #0 = { "fentry-call"="true" }
!0 = !{i32 8, !"cf-protection-branch", i32 1}
