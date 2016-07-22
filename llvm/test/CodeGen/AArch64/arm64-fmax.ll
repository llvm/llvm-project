; RUN: llc < %s -mtriple=arm64-eabi -enable-no-nans-fp-math | FileCheck %s

define double @test_direct(float %in) {
; CHECK-LABEL: test_direct:
  %cmp = fcmp nnan olt float %in, 0.000000e+00
  %val = select i1 %cmp, float 0.000000e+00, float %in
  %longer = fpext float %val to double
  ret double %longer

; CHECK: fmax
}

define double @test_cross(float %in) {
; CHECK-LABEL: test_cross:
  %cmp = fcmp nnan ult float %in, 0.000000e+00
  %val = select i1 %cmp, float %in, float 0.000000e+00
  %longer = fpext float %val to double
  ret double %longer

; CHECK: fmin
}

; Same as previous, but with ordered comparison;
; can't be converted in safe-math mode.
define double @test_cross_fail_nan(float %in) {
; CHECK-LABEL: test_cross_fail_nan:
  %cmp = fcmp nnan olt float %in, 0.000000e+00
  %val = select i1 %cmp, float %in, float 0.000000e+00
  %longer = fpext float %val to double
  ret double %longer

; CHECK: fmin
}

; This isn't a min or a max, but passes the first condition for swapping the
; results. Make sure they're put back before we resort to the normal fcsel.
define float @test_cross_fail(float %lhs, float %rhs) {
; CHECK-LABEL: test_cross_fail:
  %tst = fcmp nnan une float %lhs, %rhs
  %res = select i1 %tst, float %rhs, float %lhs
  ret float %res

  ; The register allocator would have to decide to be deliberately obtuse before
  ; other register were used.
; CHECK: fcsel s0, s1, s0, ne
}

; Make sure the transformation isn't triggered for integers
define i64 @test_integer(i64  %in) {
  %cmp = icmp slt i64 %in, 0
  %val = select i1 %cmp, i64 0, i64 %in
  ret i64 %val
}

define float @test_f16(half %in) {
; CHECK-LABEL: test_f16:
  %cmp = fcmp nnan ult half %in, 0.000000e+00
  %val = select i1 %cmp, half %in, half 0.000000e+00
  %longer = fpext half %val to float
  ret float %longer
; FIXME: It'd be nice for this to create an fmin instruction!
; CHECK: fcvt
; CHECK: fcsel
}
