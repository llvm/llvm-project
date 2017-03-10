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
