; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc-unknown-unknown < %s | FileCheck %s
define i32 @test(<4 x i32> %v, i32 %elem) #0 {
entry:
  %vecext = extractelement <4 x i32> %v, i32 %elem
  ret i32 %vecext
}
; CHECK: stxvw4x 34,
; CHECK: lwzx 3,

define float @test2(i32 signext %a) {
entry:
  %conv = bitcast i32 %a to float
  ret float %conv
}
; CHECK-NOT: mtvsr
