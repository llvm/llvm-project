; RUN: opt < %s -instsimplify -S | FileCheck %s

;; x * 0 ==> 0 when no-nans and no-signed-zero
; CHECK: mul_zero_1
define float @mul_zero_1(float %a) {
  %b = fmul nsz nnan float %a, 0.0
; CHECK: ret float 0.0
  ret float %b
}
; CHECK: mul_zero_2
define float @mul_zero_2(float %a) {
  %b = fmul fast float 0.0, %a
; CHECK: ret float 0.0
  ret float %b
}

;; x * 0 =/=> 0 when there could be nans or -0
; CHECK: no_mul_zero_1
define float @no_mul_zero_1(float %a) {
  %b = fmul nsz float %a, 0.0
; CHECK: ret float %b
  ret float %b
}
; CHECK: no_mul_zero_2
define float @no_mul_zero_2(float %a) {
  %b = fmul nnan float %a, 0.0
; CHECK: ret float %b
  ret float %b
}
; CHECK: no_mul_zero_3
define float @no_mul_zero_3(float %a) {
  %b = fmul float %a, 0.0
; CHECK: ret float %b
  ret float %b
}

; fadd [nnan ninf] X, (fsub [nnan ninf] 0, X) ==> 0
;   where nnan and ninf have to occur at least once somewhere in this
;   expression
; CHECK: fadd_fsub_0
define float @fadd_fsub_0(float %a) {
; X + -X ==> 0
  %t1 = fsub nnan ninf float 0.0, %a
  %zero1 = fadd nnan ninf float %t1, %a

  %t2 = fsub nnan float 0.0, %a
  %zero2 = fadd ninf float %t2, %a

  %t3 = fsub nnan ninf float 0.0, %a
  %zero3 = fadd float %t3, %a

  %t4 = fsub float 0.0, %a
  %zero4 = fadd nnan ninf float %t4, %a

; Dont fold this
; CHECK: %nofold = fsub float 0.0
  %nofold = fsub float 0.0, %a
; CHECK: %no_zero = fadd nnan float %nofold, %a
  %no_zero = fadd nnan float %nofold, %a

; Coalesce the folded zeros
  %zero5 = fadd float %zero1, %zero2
  %zero6 = fadd float %zero3, %zero4
  %zero7 = fadd float %zero5, %zero6

; Should get folded
  %ret = fadd nsz float %no_zero, %zero7

; CHECK: ret float %no_zero
  ret float %ret
}

; fsub nnan x, x ==> 0.0
; CHECK-LABEL: @fsub_x_x(
define float @fsub_x_x(float %a) {
; X - X ==> 0
  %zero1 = fsub nnan float %a, %a

; Dont fold
; CHECK: %no_zero1 = fsub
  %no_zero1 = fsub ninf float %a, %a
; CHECK: %no_zero2 = fsub
  %no_zero2 = fsub float %a, %a
; CHECK: %no_zero = fadd
  %no_zero = fadd float %no_zero1, %no_zero2

; Should get folded
  %ret = fadd nsz float %no_zero, %zero1

; CHECK: ret float %no_zero
  ret float %ret
}

; fadd nsz X, 0 ==> X
; CHECK-LABEL: @nofold_fadd_x_0(
define float @nofold_fadd_x_0(float %a) {
; Dont fold
; CHECK: %no_zero1 = fadd
  %no_zero1 = fadd ninf float %a, 0.0
; CHECK: %no_zero2 = fadd
  %no_zero2 = fadd nnan float %a, 0.0
; CHECK: %no_zero = fadd
  %no_zero = fadd float %no_zero1, %no_zero2

; CHECK: ret float %no_zero
  ret float %no_zero
}

; fdiv nsz nnan 0, X ==> 0
define double @fdiv_zero_by_x(double %X) {
; CHECK-LABEL: @fdiv_zero_by_x(
; 0 / X -> 0
  %r = fdiv nnan nsz double 0.0, %X
  ret double %r
; CHECK: ret double 0
}

define float @fdiv_self(float %f) {
  %div = fdiv nnan float %f, %f
  ret float %div
; CHECK-LABEL: fdiv_self
; CHECK: ret float 1.000000e+00
}

define float @fdiv_self_invalid(float %f) {
  %div = fdiv float %f, %f
  ret float %div
; CHECK-LABEL: fdiv_self_invalid
; CHECK: %div = fdiv float %f, %f
; CHECK-NEXT: ret float %div
}

define float @fdiv_neg1(float %f) {
  %neg = fsub fast float -0.000000e+00, %f
  %div = fdiv nnan float %neg, %f
  ret float %div
; CHECK-LABEL: fdiv_neg1
; CHECK: ret float -1.000000e+00
}

define float @fdiv_neg2(float %f) {
  %neg = fsub fast float 0.000000e+00, %f
  %div = fdiv nnan float %neg, %f
  ret float %div
; CHECK-LABEL: fdiv_neg2
; CHECK: ret float -1.000000e+00
}

define float @fdiv_neg_invalid(float %f) {
  %neg = fsub fast float -0.000000e+00, %f
  %div = fdiv float %neg, %f
  ret float %div
; CHECK-LABEL: fdiv_neg_invalid
; CHECK: %neg = fsub fast float -0.000000e+00, %f
; CHECK-NEXT: %div = fdiv float %neg, %f
; CHECK-NEXT: ret float %div
}

define float @fdiv_neg_swapped1(float %f) {
  %neg = fsub float -0.000000e+00, %f
  %div = fdiv nnan float %f, %neg
  ret float %div
; CHECK-LABEL: fdiv_neg_swapped1
; CHECK: ret float -1.000000e+00
}

define float @fdiv_neg_swapped2(float %f) {
  %neg = fsub float 0.000000e+00, %f
  %div = fdiv nnan float %f, %neg
  ret float %div
; CHECK-LABEL: fdiv_neg_swapped2
; CHECK: ret float -1.000000e+00
}
