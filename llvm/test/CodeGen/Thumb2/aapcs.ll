; RUN: llc < %s -mtriple=thumbv7-none-eabi   -mcpu=cortex-m4 -mattr=-vfp2             | FileCheck %s -check-prefix=CHECK -check-prefix=SOFT
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m4 -mattr=+vfp4,-fp64 | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=SP
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-a8 -mattr=+vfp3             | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=DP

define float @float_in_reg(float %a, float %b) {
entry:
; CHECK-LABEL: float_in_reg:
; SOFT: mov r0, r1
; HARD: vmov.f32  s0, s1
; CHECK-NEXT: bx lr
  ret float %b
}

define double @double_in_reg(double %a, double %b) {
entry:
; CHECK-LABEL: double_in_reg:
; SOFT: mov r1, r3
; SOFT: mov r0, r2
; SP: vmov.f32  s0, s2
; SP: vmov.f32  s1, s3
; DP: vmov.f64  d0, d1
; CHECK-NEXT: bx lr
  ret double %b
}

define float @float_on_stack(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, float %i) {
; CHECK-LABEL: float_on_stack:
; SOFT: ldr r0, [sp, #48]
; HARD: vldr s0, [sp]
; CHECK-NEXT: bx lr
  ret float %i
}

define double @double_on_stack(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i) {
; CHECK-LABEL: double_on_stack:
; SOFT: ldrd r0, r1, [sp, #48]
; HARD: vldr d0, [sp]
; CHECK-NEXT: bx lr
  ret double %i
}

define double @double_not_split(double %a, double %b, double %c, double %d, double %e, double %f, double %g, float %h, double %i) {
; CHECK-LABEL: double_not_split:
; SOFT: ldrd r0, r1, [sp, #48]
; HARD: vldr d0, [sp]
; CHECK-NEXT: bx lr
  ret double %i
}
