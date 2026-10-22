; RUN: opt -S -passes=hipstdpar-math-fixup %s | FileCheck %s

; Library functions must only be redirected to their mapped replacement, and
; must not also be treated as intrinsics, which would insert bogus
; declarations derived from the function name.

; CHECK-LABEL: define void @test(
; CHECK: call double @__hipstdpar_acosh_f64(
; CHECK: call float @__hipstdpar_remainder_f32(
; CHECK: declare hidden float @remainderf(float, float)
; CHECK-EMPTY:
; CHECK-NEXT: declare double @__hipstdpar_acosh_f64(double)
; CHECK-EMPTY:
; CHECK-NEXT: declare float @__hipstdpar_remainder_f32(float, float)

define void @test(double %dbl, float %flt) {
entry:
  %0 = call double @acosh(double %dbl)
  %1 = call float @remainderf(float %flt, float %flt)
  ret void
}

declare hidden double @acosh(double)

declare hidden float @remainderf(float, float)
