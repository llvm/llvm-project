; RUN: llc %s -mtriple=x86_64-linux-gnu -o - -verify-machineinstrs \
; RUN:   | FileCheck %s

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: test1:
; CHECK: .section __mcount_loc,"a",@progbits
; CHECK: .quad .Ltmp0
; CHECK: .text
; CHECK: .Ltmp0:
; CHECK: callq __fentry__
; CHECK: retq 
}

define void @test2() #1 {
entry:
  ret void

; CHECK-LABEL: test2:
; CHECK: .section __mcount_loc,"a",@progbits
; CHECK: .quad .Ltmp1
; CHECK: .text
; CHECK: .Ltmp1:
; CHECK: nopl 8(%rax,%rax)
; CHECK-NOT: callq __fentry__
; CHECK: retq
}

attributes #0 = { "fentry-call"="true" "mrecord-mcount" }
attributes #1 = { "fentry-call"="true" "mnop-mcount" "mrecord-mcount" }
