; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; Check that we convert
;   trunc(C * a) -> trunc(C) * trunc(a)
; if C is a constant.
; CHECK-LABEL: @trunc_of_mul
define i8 @trunc_of_mul(i32 %a) {
  %b = mul i32 %a, 100
  ; CHECK: %c
  ; CHECK-NEXT: --> (100 * (trunc i32 %a to i8))
  %c = trunc i32 %b to i8
  ret i8 %c
}

; Check that we convert
;   trunc(C + a) -> trunc(C) + trunc(a)
; if C is a constant.
; CHECK-LABEL: @trunc_of_add
define i8 @trunc_of_add(i32 %a) {
  %b = add i32 %a, 100
  ; CHECK: %c
  ; CHECK-NEXT: --> (100 + (trunc i32 %a to i8))
  %c = trunc i32 %b to i8
  ret i8 %c
}
