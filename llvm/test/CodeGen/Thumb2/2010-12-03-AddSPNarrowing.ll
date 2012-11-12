; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s
; Radar 8724703: Make sure that a t2ADDrSPi instruction with SP as the
; destination register is narrowed to tADDspi instead of tADDrSPi.

define void @test() nounwind {
entry:
; CHECK: sub.w
; CHECK: add.w
  %Buffer.i = alloca [512 x i8], align 4
  ret void
}
