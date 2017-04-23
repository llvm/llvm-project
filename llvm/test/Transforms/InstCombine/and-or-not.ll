; RUN: opt < %s -instcombine -S | FileCheck %s

; PR1510

; (a | b) & ~(a & b) --> a ^ b

define i32 @and_to_xor(i32 %a, i32 %b) {
; CHECK-LABEL: @and_to_xor(
; CHECK-NEXT:    [[AND2:%.*]] = xor i32 %b, %a
; CHECK-NEXT:    ret i32 [[AND2]]
;
  %or = or i32 %b, %a
  %and = and i32 %b, %a
  %not = xor i32 %and, -1
  %and2 = and i32 %or, %not
  ret i32 %and2
}

define <4 x i32> @and_to_xor_vec(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @and_to_xor_vec(
; CHECK-NEXT:    [[AND2:%.*]] = xor <4 x i32> %a, %b
; CHECK-NEXT:    ret <4 x i32> [[AND2]]
;
  %or = or <4 x i32> %a, %b
  %and = and <4 x i32> %a, %b
  %not = xor <4 x i32> %and, < i32 -1, i32 -1, i32 -1, i32 -1 >
  %and2 = and <4 x i32> %or, %not
  ret <4 x i32> %and2
}

; ~(~(a | b) | (a & b)) --> (a | b) & ~(a & b) -> a ^ b

define i32 @demorgan_plus_and_to_xor(i32 %a, i32 %b) {
; CHECK-LABEL: @demorgan_plus_and_to_xor(
; CHECK-NEXT:    [[NOT:%.*]] = xor i32 %b, %a
; CHECK-NEXT:    ret i32 [[NOT]]
;
  %or = or i32 %b, %a
  %notor = xor i32 %or, -1
  %and = and i32 %b, %a
  %or2 = or i32 %and, %notor
  %not = xor i32 %or2, -1
  ret i32 %not
}

define <4 x i32> @demorgan_plus_and_to_xor_vec(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @demorgan_plus_and_to_xor_vec(
; CHECK-NEXT:    [[NOT:%.*]] = xor <4 x i32> %a, %b
; CHECK-NEXT:    ret <4 x i32> [[NOT]]
;
  %or = or <4 x i32> %a, %b
  %notor = xor <4 x i32> %or, < i32 -1, i32 -1, i32 -1, i32 -1 >
  %and = and <4 x i32> %a, %b
  %or2 = or <4 x i32> %and, %notor
  %not = xor <4 x i32> %or2, < i32 -1, i32 -1, i32 -1, i32 -1 >
  ret <4 x i32> %not
}

