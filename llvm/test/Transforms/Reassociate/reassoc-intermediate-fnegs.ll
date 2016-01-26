; RUN: opt < %s -reassociate -S | FileCheck %s
; CHECK-LABEL: faddsubAssoc1
; CHECK: [[TMP1:%.*]] = fsub fast half 0xH8000, %a
; CHECK: [[TMP2:%.*]] = fadd fast half %b, [[TMP1]]
; CHECK: fmul fast half [[TMP2]], 0xH4500
; CHECK: ret
; Input is A op (B op C)
define half @faddsubAssoc1(half %a, half %b) {
  %tmp1 = fmul fast half %b, 0xH4200 ; 3*b
  %tmp2 = fmul fast half %a, 0xH4500 ; 5*a
  %tmp3 = fmul fast half %b, 0xH4000 ; 2*b
  %tmp4 = fsub fast half %tmp2, %tmp1 ; 5 * a - 3 * b
  %tmp5 = fsub fast half %tmp3, %tmp4 ; 2 * b - ( 5 * a - 3 * b)
  ret half %tmp5 ; = 5 * (b - a)
}

; CHECK-LABEL: faddsubAssoc2
; CHECK: [[TMP1:%tmp.*]] = fmul fast half %a, 0xH4500
; CHECK: [[TMP2:%tmp.*]] = fmul fast half %b, 0xH3C00
; CHECK: fadd fast half [[TMP2]], [[TMP1]]
; CHECK: ret
; Input is (A op B) op C
define half @faddsubAssoc2(half %a, half %b) {
  %tmp1 = fmul fast half %b, 0xH4200 ; 3*b
  %tmp2 = fmul fast half %a, 0xH4500 ; 5*a
  %tmp3 = fmul fast half %b, 0xH4000 ; 2*b
  %tmp4 = fadd fast half %tmp2, %tmp1 ; 5 * a + 3 * b
  %tmp5 = fsub fast half %tmp4, %tmp3 ; (5 * a + 3 * b) - (2 * b)
  ret half %tmp5 ; = 5 * a + b
}

