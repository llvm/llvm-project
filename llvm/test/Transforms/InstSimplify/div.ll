; RUN: opt < %s -instsimplify -S | FileCheck %s

; Division-by-zero is undef. UB in any vector lane means the whole op is undef.

define <2 x i8> @sdiv_zero_elt_vec_constfold(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_zero_elt_vec_constfold(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %div = sdiv <2 x i8> <i8 1, i8 2>, <i8 0, i8 -42>
  ret <2 x i8> %div
}

define <2 x i8> @udiv_zero_elt_vec_constfold(<2 x i8> %x) {
; CHECK-LABEL: @udiv_zero_elt_vec_constfold(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %div = udiv <2 x i8> <i8 1, i8 2>, <i8 42, i8 0>
  ret <2 x i8> %div
}

define <2 x i8> @sdiv_zero_elt_vec(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_zero_elt_vec(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %div = sdiv <2 x i8> %x, <i8 -42, i8 0>
  ret <2 x i8> %div
}

define <2 x i8> @udiv_zero_elt_vec(<2 x i8> %x) {
; CHECK-LABEL: @udiv_zero_elt_vec(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %div = udiv <2 x i8> %x, <i8 0, i8 42>
  ret <2 x i8> %div
}

define <2 x i8> @sdiv_undef_elt_vec(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_undef_elt_vec(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %div = sdiv <2 x i8> %x, <i8 -42, i8 undef>
  ret <2 x i8> %div
}

define <2 x i8> @udiv_undef_elt_vec(<2 x i8> %x) {
; CHECK-LABEL: @udiv_undef_elt_vec(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %div = udiv <2 x i8> %x, <i8 undef, i8 42>
  ret <2 x i8> %div
}

; Division-by-zero is undef. UB in any vector lane means the whole op is undef.
; Thus, we can simplify this: if any element of 'y' is 0, we can do anything.
; Therefore, assume that all elements of 'y' must be 1.

define <2 x i1> @sdiv_bool_vec(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @sdiv_bool_vec(
; CHECK-NEXT:    ret <2 x i1> %x
;
  %div = sdiv <2 x i1> %x, %y
  ret <2 x i1> %div
}

define <2 x i1> @udiv_bool_vec(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @udiv_bool_vec(
; CHECK-NEXT:    ret <2 x i1> %x
;
  %div = udiv <2 x i1> %x, %y
  ret <2 x i1> %div
}

define i32 @udiv_dividend_known_smaller_than_constant_divisor(i32 %x) {
; CHECK-LABEL: @udiv_dividend_known_smaller_than_constant_divisor(
; CHECK-NEXT:    ret i32 0
;
  %and = and i32 %x, 250
  %div = udiv i32 %and, 251
  ret i32 %div
}

define i32 @not_udiv_dividend_known_smaller_than_constant_divisor(i32 %x) {
; CHECK-LABEL: @not_udiv_dividend_known_smaller_than_constant_divisor(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 251
; CHECK-NEXT:    [[DIV:%.*]] = udiv i32 [[AND]], 251
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %and = and i32 %x, 251
  %div = udiv i32 %and, 251
  ret i32 %div
}

define i32 @udiv_constant_dividend_known_smaller_than_divisor(i32 %x) {
; CHECK-LABEL: @udiv_constant_dividend_known_smaller_than_divisor(
; CHECK-NEXT:    ret i32 0
;
  %or = or i32 %x, 251
  %div = udiv i32 250, %or
  ret i32 %div
}

define i32 @not_udiv_constant_dividend_known_smaller_than_divisor(i32 %x) {
; CHECK-LABEL: @not_udiv_constant_dividend_known_smaller_than_divisor(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, 251
; CHECK-NEXT:    [[DIV:%.*]] = udiv i32 251, [[OR]]
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %or = or i32 %x, 251
  %div = udiv i32 251, %or
  ret i32 %div
}

; This would require computing known bits on both x and y. Is it worth doing?

define i32 @udiv_dividend_known_smaller_than_divisor(i32 %x, i32 %y) {
; CHECK-LABEL: @udiv_dividend_known_smaller_than_divisor(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 250
; CHECK-NEXT:    [[OR:%.*]] = or i32 %y, 251
; CHECK-NEXT:    [[DIV:%.*]] = udiv i32 [[AND]], [[OR]]
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %and = and i32 %x, 250
  %or = or i32 %y, 251
  %div = udiv i32 %and, %or
  ret i32 %div
}

define i32 @not_udiv_dividend_known_smaller_than_divisor(i32 %x, i32 %y) {
; CHECK-LABEL: @not_udiv_dividend_known_smaller_than_divisor(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 251
; CHECK-NEXT:    [[OR:%.*]] = or i32 %y, 251
; CHECK-NEXT:    [[DIV:%.*]] = udiv i32 [[AND]], [[OR]]
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %and = and i32 %x, 251
  %or = or i32 %y, 251
  %div = udiv i32 %and, %or
  ret i32 %div
}

declare i32 @external()

define i32 @div1() {
; CHECK-LABEL: @div1(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @external(), !range !0
; CHECK-NEXT:    ret i32 0
;
  %call = call i32 @external(), !range !0
  %urem = udiv i32 %call, 3
  ret i32 %urem
}

!0 = !{i32 0, i32 3}
