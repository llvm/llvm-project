; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define double @test_direct(float %in) {
; CHECK-LABEL: test_direct:
  %cmp = fcmp olt float %in, 0.000000e+00
  %val = select i1 %cmp, float 0.000000e+00, float %in
  %longer = fpext float %val to double
  ret double %longer

; CHECK: fmax s
}

define double @test_cross(float %in) {
; CHECK-LABEL: test_cross:
  %cmp = fcmp ult float %in, 0.000000e+00
  %val = select i1 %cmp, float %in, float 0.000000e+00
  %longer = fpext float %val to double
  ret double %longer

; CHECK: fmin s
}

; Same as previous, but with ordered comparison;
; must become fminnm, not fmin.
define double @test_cross_fail_nan(float %in) {
; CHECK-LABEL: test_cross_fail_nan:
  %cmp = fcmp olt float %in, 0.000000e+00
  %val = select i1 %cmp, float %in, float 0.000000e+00
  %longer = fpext float %val to double
  ret double %longer

; CHECK: fminnm s
}

; This isn't a min or a max, but passes the first condition for swapping the
; results. Make sure they're put back before we resort to the normal fcsel.
define float @test_cross_fail(float %lhs, float %rhs) {
; CHECK-LABEL: test_cross_fail:
  %tst = fcmp une float %lhs, %rhs
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
