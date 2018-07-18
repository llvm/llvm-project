; RUN: llc -mtriple=thumbv7m-apple-darwin -mcpu=cortex-m3 < %s | FileCheck %s --check-prefix=CHECK-M3
; RUN: llc -mtriple=thumbv7em-apple-darwin -mcpu=cortex-m4 < %s | FileCheck %s --check-prefix=CHECK-M4
; RUN: llc -mtriple=thumbv7-apple-darwin -mcpu=cortex-m3 < %s | FileCheck %s --check-prefix=CHECK-M3
; RUN: llc -mtriple=thumbv7-apple-darwin -mcpu=cortex-m4 < %s | FileCheck %s --check-prefix=CHECK-M4

define float @float_op(float %lhs, float %rhs) {
  %sum = fadd float %lhs, %rhs
  ret float %sum
; CHECK-M3-LABEL: float_op:
; CHECK-M3: bl ___addsf3

; CHECK-M4-LABEL: float_op:
; CHECK-M4: vadd.f32
}

define double @double_op(double %lhs, double %rhs) {
  %sum = fadd double %lhs, %rhs
  ret double %sum
; CHECK-M3-LABEL: double_op:
; CHECK-M3: bl ___adddf3

; CHECK-M4-LABEL: double_op:
; CHECK-M4: {{(bl|blx|b.w)}} ___adddf3
}
