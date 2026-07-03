; RUN: llc %s -mtriple=sparc -o - | FileCheck --check-prefix=CHECK --check-prefix=DEFAULT %s
; RUN: llc %s -mtriple=sparc -mattr=no-fmuls -o - | FileCheck --check-prefix=CHECK --check-prefix=NO-FMULS %s
; RUN: llc %s -mtriple=sparc -mattr=no-fsmuld -o - | FileCheck --check-prefix=CHECK --check-prefix=NO-FSMULD %s
; RUN: llc %s -mtriple=sparc -mattr=no-fsmuld,no-fmuls -o - | FileCheck --check-prefix=CHECK --check-prefix=NO-BOTH %s

;;; Test case ensures that the no-fsmuld and no-fmuls features disable
;;; the relevant instruction, and alternative sequences get emitted
;;; instead.

; CHECK-LABEL: test_float_mul:
; DEFAULT:     fmuls
; NO-FSMULD:   fmuls
; NO-FMULS:    fsmuld
; NO-FMULS:    fdtos
; NO-BOTH:     fstod
; NO-BOTH:     fstod
; NO-BOTH:     fmuld
; NO-BOTH:     fdtos
define float @test_float_mul(float %a, float %b) {
entry:
  %mul = fmul float %a, %b

  ret float %mul
}

; CHECK-LABEL: test_float_mul_double:
; DEFAULT:     fsmuld
; NO-FSMULD:   fstod
; NO-FSMULD:   fstod
; NO-FSMULD:   fmuld
define double @test_float_mul_double(float %a, float %b) {
entry:
  %a_double = fpext float %a to double
  %b_double = fpext float %b to double
  %mul = fmul double %a_double, %b_double

  ret double %mul
}
