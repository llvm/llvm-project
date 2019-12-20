; RUN: llc %s -mtriple=s390x-linux-gnu -mcpu=z10 -o - -verify-machineinstrs \
; RUN:   | FileCheck %s

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: test1:
; CHECK: .section __mcount_loc,"a",@progbits
; CHECK: .quad .Ltmp0
; CHECK: .text
; CHECK: .Ltmp0:
; CHECK: brasl %r0, __fentry__@PLT
; CHECK: br %r14
}

define void @test2() #1 {
entry:
  ret void

; CHECK-LABEL: test2:
; CHECK: .section __mcount_loc,"a",@progbits
; CHECK: .quad .Ltmp1
; CHECK: .text
; CHECK: .Ltmp1:
; CHECK: brcl 0, .Ltmp2
; CHECK: .Ltmp2:
; CHECK: br %r14
}

attributes #0 = { "fentry-call"="true" "mrecord-mcount" }
attributes #1 = { "fentry-call"="true" "mnop-mcount" "mrecord-mcount" }
