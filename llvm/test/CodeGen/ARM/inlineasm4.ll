; RUN: llc < %s -march=arm | FileCheck %s

define double @f(double %x) {
entry:
  %0 = tail call double asm "mov     ${0:R}, #4\0A", "=&r"()
  ret double %0
; CHECK: f:
; CHECK:	mov     r1, #4
}

define double @g(double %x) {
entry:
  %0 = tail call double asm "mov     ${0:Q}, #4\0A", "=&r"()
  ret double %0
; CHECK: g:
; CHECK:	mov     r0, #4
}
