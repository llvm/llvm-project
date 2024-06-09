; Check that we can enable/disable UnsafeFPMath via function attributes.  An
; attribute on one function should not magically apply to the next one.

; RUN: llc < %s -mtriple=x86_64-unknown-unknown \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=SAFE

; RUN: llc < %s -mtriple=x86_64-unknown-unknown -enable-unsafe-fp-math \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=UNSAFE

; The div in these functions should be converted to a mul when unsafe-fp-math
; is enabled.

; CHECK-LABEL: unsafe_fp_math_default0:
define double @unsafe_fp_math_default0(double %x) {
; SAFE:      divsd
; UNSAFE:    mulsd
  %div = fdiv double %x, 3.0
  ret double %div
}

; CHECK-LABEL: unsafe_fp_math_off:
define double @unsafe_fp_math_off(double %x) #0 {
; SAFE:      divsd
; UNSAFE:    divsd
  %div = fdiv double %x, 3.0
  ret double %div
}

; CHECK-LABEL: unsafe_fp_math_default1:
define double @unsafe_fp_math_default1(double %x) {
; With unsafe math enabled, can change this div to a mul.
; SAFE:      divsd
; UNSAFE:    mulsd
  %div = fdiv double %x, 3.0
  ret double %div
}

; CHECK-LABEL: unsafe_fp_math_on:
define double @unsafe_fp_math_on(double %x) #1 {
; SAFE:      mulsd
; UNSAFE:    mulsd
  %div = fdiv double %x, 3.0
  ret double %div
}

; CHECK-LABEL: unsafe_fp_math_default2:
define double @unsafe_fp_math_default2(double %x) {
; With unsafe math enabled, can change this div to a mul.
; SAFE:      divsd
; UNSAFE:    mulsd
  %div = fdiv double %x, 3.0
  ret double %div
}

attributes #0 = { "unsafe-fp-math"="false" }
attributes #1 = { "unsafe-fp-math"="true" }
