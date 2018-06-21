; RUN: opt < %s -instcombine -S | FileCheck %s

; PR4374
define float @test1(float %a, float %b) nounwind {
  %t1 = fsub float %a, %b
  %t2 = fsub float -0.000000e+00, %t1

; CHECK:       %t1 = fsub float %a, %b
; CHECK-NEXT:  %t2 = fsub float -0.000000e+00, %t1

  ret float %t2
}

; <rdar://problem/7530098>
define double @test2(double %x, double %y) nounwind {
  %t1 = fadd double %x, %y
  %t2 = fsub double %x, %t1

; CHECK:      %t1 = fadd double %x, %y
; CHECK-NEXT: %t2 = fsub double %x, %t1

  ret double %t2
}

; CHECK-LABEL: @fsub_undef(
; CHECK: %sub = fsub float %val, undef
define float @fsub_undef(float %val) {
bb:
  %sub = fsub float %val, undef
  ret float %sub
}

; XXX - Why doesn't this fold to undef?
; CHECK-LABEL: @fsub_fast_undef(
; CHCK: %sub = fsub fast float %val, undef
define float @fsub_fast_undef(float %val) {
bb:
  %sub = fsub fast float %val, undef
  ret float %sub
}

; CHECK-LABEL: @fneg_undef(
; CHECK: ret float fsub (float -0.000000e+00, float undef)
define float @fneg_undef(float %val) {
bb:
  %sub = fsub float -0.0, undef
  ret float %sub
}

; CHECK-LABEL: @fneg_fast_undef(
; CHECK: ret float fsub (float -0.000000e+00, float undef)
define float @fneg_fast_undef(float %val) {
bb:
  %sub = fsub fast float -0.0, undef
  ret float %sub
}

; This folds to a constant expression, which produced 0 instructions
; contrary to the expected one for negation.
; CHECK-LABEL: @inconsistent_numbers_fsub_undef(
; CHECK: ret float fsub (float -0.000000e+00, float undef)
define float @inconsistent_numbers_fsub_undef(float %val) {
bb:
  %sub0 = fsub fast float %val, undef
  %sub1 = fsub fast float %sub0, %val
  ret float %sub1
}
