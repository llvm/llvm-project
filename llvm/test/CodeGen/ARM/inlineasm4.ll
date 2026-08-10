; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define double @f(double %x) {
entry:
  %0 = tail call double asm "mov     ${0:R}, #4\0A", "=&r"()
  ret double %0
; CHECK-LABEL: f:
; CHECK:	mov     r1, #4
}

define double @g(double %x) {
entry:
  %0 = tail call double asm "mov     ${0:Q}, #4\0A", "=&r"()
  ret double %0
; CHECK-LABEL: g:
; CHECK:	mov     r0, #4
}
