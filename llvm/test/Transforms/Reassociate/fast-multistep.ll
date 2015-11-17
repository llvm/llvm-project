; RUN: opt < %s -reassociate -S | FileCheck %s

define float @fmultistep1(float %a, float %b, float %c) {
; Check that a*a*b+a*a*c is turned into a*(a*(b+c)).
; CHECK-LABEL: @fmultistep1
; CHECK-NEXT: [[TMP1:%tmp.*]] = fadd fast float %c, %b
; CHECK-NEXT: [[TMP2:%tmp.*]] = fmul fast float %a, %a
; CHECK-NEXT: fmul fast float [[TMP2]], [[TMP1]]
; CHECK-NEXT: ret float

  %t0 = fmul fast float %a, %b
  %t1 = fmul fast float %a, %t0 ; a*(a*b)
  %t2 = fmul fast float %a, %c
  %t3 = fmul fast float %a, %t2 ; a*(a*c)
  %t4 = fadd fast float %t1, %t3
  ret float %t4
}

define float @fmultistep2(float %a, float %b, float %c, float %d) {
; Check that a*b+a*c+d is turned into a*(b+c)+d.
; CHECK-LABEL: @fmultistep2
; CHECK-NEXT: fadd fast float %c, %b
; CHECK-NEXT: fmul fast float %tmp, %a
; CHECK-NEXT: fadd fast float %tmp1, %d
; CHECK-NEXT: ret float

  %t0 = fmul fast float %a, %b
  %t1 = fmul fast float %a, %c
  %t2 = fadd fast float %t1, %d ; a*c+d
  %t3 = fadd fast float %t0, %t2 ; a*b+(a*c+d)
  ret float %t3
}
