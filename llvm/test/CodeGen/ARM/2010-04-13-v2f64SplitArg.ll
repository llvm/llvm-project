; RUN: llc < %s -mtriple=arm-apple-darwin -mcpu=cortex-a8
; Radar 7855014

define void @test1(i32 %f0, i32 %f1, i32 %f2, <4 x i32> %f3) nounwind {
entry:
  unreachable
}
