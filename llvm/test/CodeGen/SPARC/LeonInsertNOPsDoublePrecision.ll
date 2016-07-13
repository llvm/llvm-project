; RUN: llc %s -O0 -march=sparc -mcpu=at697f -o - | FileCheck %s

; CHECK:        ldd
; CHECK:        ldd
; CHECK-NEXT:   nop

define double @mult() #0 {
entry:
  %x = alloca double, align 8
  %y = alloca double, align 8
  store double 3.141590e+00, double* %x, align 8
  store double 1.234560e+00, double* %y, align 8
  %0 = load double, double* %x, align 8
  %1 = load double, double* %y, align 8
  %mul = fmul double %0, %1
  ret double %mul
}
