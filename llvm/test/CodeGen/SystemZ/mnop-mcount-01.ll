; RUN: llc %s -mtriple=s390x-linux-gnu -mcpu=z10 -o - -verify-machineinstrs \
; RUN:   | FileCheck %s

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: @test1
; CHECK: brasl %r0, __fentry__
; CHECK-NOT: brcl 0, .Ltmp0
; CHECK: br %r14
}

define void @test2() #1 {
entry:
  ret void

; CHECK-LABEL: @test2
; CHECK-NOT: brasl %r0, __fentry__
; CHECK: brcl 0, .Ltmp0
; CHECK: br %r14
}

attributes #0 = { "fentry-call"="true" }
attributes #1 = { "fentry-call"="true" "mnop-mcount" }

