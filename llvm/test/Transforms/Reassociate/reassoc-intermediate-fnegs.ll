; RUN: opt < %s -reassociate -S | FileCheck %s

; Input is A op (B op C)

define half @faddsubAssoc1(half %a, half %b) {
; CHECK-LABEL: @faddsubAssoc1(
; CHECK-NEXT:    [[T2_NEG:%.*]] = fmul fast half %a, 0xH4500
; CHECK-NEXT:    [[REASS_MUL:%.*]] = fmul fast half %b, 0xH4500
; CHECK-NEXT:    [[T51:%.*]] = fsub fast half [[REASS_MUL]], [[T2_NEG]]
; CHECK-NEXT:    [[T5:%.*]] = fadd fast half [[REASS_MUL]], [[T2_NEG]]
; CHECK-NEXT:    ret half [[T51]]
;
  %t1 = fmul fast half %b, 0xH4200 ; 3*b
  %t2 = fmul fast half %a, 0xH4500 ; 5*a
  %t3 = fmul fast half %b, 0xH4000 ; 2*b
  %t4 = fsub fast half %t2, %t1 ; 5 * a - 3 * b
  %t5 = fsub fast half %t3, %t4 ; 2 * b - ( 5 * a - 3 * b)
  ret half %t5 ; = 5 * (b - a)
}

; Input is (A op B) op C

define half @faddsubAssoc2(half %a, half %b) {
; CHECK-LABEL: @faddsubAssoc2(
; CHECK-NEXT:    [[T2:%.*]] = fmul fast half %a, 0xH4500
; CHECK-NEXT:    [[REASS_MUL:%.*]] = fmul fast half %b, 0xH3C00
; CHECK-NEXT:    [[T5:%.*]] = fadd fast half [[REASS_MUL]], [[T2]]
; CHECK-NEXT:    ret half [[T5]]
;
  %t1 = fmul fast half %b, 0xH4200 ; 3*b
  %t2 = fmul fast half %a, 0xH4500 ; 5*a
  %t3 = fmul fast half %b, 0xH4000 ; 2*b
  %t4 = fadd fast half %t2, %t1 ; 5 * a + 3 * b
  %t5 = fsub fast half %t4, %t3 ; (5 * a + 3 * b) - (2 * b)
  ret half %t5 ; = 5 * a + b
}

