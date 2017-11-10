; RUN: opt < %s -reassociate -S | FileCheck %s

; Check that a*a*b+a*a*c is turned into a*(a*(b+c)).

define i64 @multistep1(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: @multistep1(
; CHECK-NEXT:    [[REASS_ADD1:%.*]] = add i64 %c, %b
; CHECK-NEXT:    [[REASS_MUL2:%.*]] = mul i64 %a, %a
; CHECK-NEXT:    [[REASS_MUL:%.*]] = mul i64 [[REASS_MUL:%.*]]2, [[REASS_ADD1]]
; CHECK-NEXT:    ret i64 [[REASS_MUL]]
;
  %t0 = mul i64 %a, %b
  %t1 = mul i64 %a, %t0 ; a*(a*b)
  %t2 = mul i64 %a, %c
  %t3 = mul i64 %a, %t2 ; a*(a*c)
  %t4 = add i64 %t1, %t3
  ret i64 %t4
}

; Check that a*b+a*c+d is turned into a*(b+c)+d.

define i64 @multistep2(i64 %a, i64 %b, i64 %c, i64 %d) {
; CHECK-LABEL: @multistep2(
; CHECK-NEXT:    [[REASS_ADD:%.*]] = add i64 %c, %b
; CHECK-NEXT:    [[REASS_MUL:%.*]] = mul i64 [[REASS_ADD]], %a
; CHECK-NEXT:    [[T3:%.*]] = add i64 [[REASS_MUL]], %d
; CHECK-NEXT:    ret i64 [[T3]]
;
  %t0 = mul i64 %a, %b
  %t1 = mul i64 %a, %c
  %t2 = add i64 %t1, %d ; a*c+d
  %t3 = add i64 %t0, %t2 ; a*b+(a*c+d)
  ret i64 %t3
}

