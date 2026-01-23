; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define double @test_direct(float %in) {
; CHECK-LABEL: test_direct:
  %cmp = fcmp olt float %in, 0.000000e+00
  %val = select i1 %cmp, float 0.000000e+00, float %in
  %longer = fpext float %val to double
  ret double %longer

; CHECK: fcmp
; CHECK: fcsel
}

define double @test_cross(float %in) {
; CHECK-LABEL: test_cross:
  %cmp = fcmp ult float %in, 0.000000e+00
  %val = select i1 %cmp, float %in, float 0.000000e+00
  %longer = fpext float %val to double
  ret double %longer

; CHECK: fcmp
; CHECK: fcsel
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

; Make sure we don't translate it into fminnm when the nsz flag is set on the fcmp.
define float @minnum_fcmp_nsz(float %x, float %y) {
; CHECK-LABEL: minnum_fcmp_nsz:
  %cmp = fcmp nnan nsz ole float %x, %y
  %sel = select i1 %cmp, float %x, float %y
  ret float %sel
; CHECK-NOT: fminnm
; CHECK: fcsel s0, s0, s1, le
}

; Make sure we translate it into fminnm when the nsz flag is set on the select.
define float @minnum_select_nsz(float %x, float %y) {
; CHECK-LABEL: minnum_select_nsz:
  %cmp = fcmp nnan ole float %x, %y
  %sel = select nsz i1 %cmp, float %x, float %y
  ret float %sel
; CHECK: fminnm s0, s0, s1
}
