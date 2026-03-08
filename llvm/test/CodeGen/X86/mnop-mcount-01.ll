; RUN: llc %s -mtriple=x86_64-linux-gnu -o - -verify-machineinstrs \
; RUN:   | FileCheck %s

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: @test1
; CHECK: callq __fentry__
; CHECK-NOT: nopl 8(%rax,%rax)
; CHECK: retq
}

define void @test2() #1 {
entry:
  ret void

; CHECK-LABEL: @test2
; CHECK-NOT: callq __fentry__
; CHECK: nopl 8(%rax,%rax)
; CHECK: retq
}

attributes #0 = { "fentry-call"="true" }
attributes #1 = { "fentry-call"="true" "mnop-mcount" }

