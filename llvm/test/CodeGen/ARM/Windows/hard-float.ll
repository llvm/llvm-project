; RUN: llc -mtriple=thumbv7-windows-itanium -mcpu=cortex-a9 -o - %s \
; RUN:   | FileCheck %s -check-prefix CHECK-WIN

; RUN: llc -mtriple=thumbv7-windows-gnu -mcpu=cortex-a9 -o - %s \
; RUN:   | FileCheck %s -check-prefix CHECK-GNU

define float @function(float %f, float %g) nounwind {
entry:
  %h = fadd float %f, %g
  ret float %h
}

; CHECK-WIN: vadd.f32 s0, s0, s1

; CHECK-GNU: vadd.f32 s0, s0, s1

