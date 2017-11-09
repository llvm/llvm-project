; RUN: opt -reassociate %s -S | FileCheck %s

define float @foo(float %a,float %b, float %c) {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    [[MUL3:%.*]] = fmul float %a, %b
; CHECK-NEXT:    [[FACTOR:%.*]] = fmul fast float %c, 2.000000e+00
; CHECK-NEXT:    [[REASS_ADD1:%.*]] = fadd fast float [[FACTOR]], %b
; CHECK-NEXT:    [[REASS_MUL:%.*]] = fmul fast float [[REASS_ADD1]], %a
; CHECK-NEXT:    [[ADD3:%.*]] = fadd fast float [[REASS_MUL]], [[MUL3]]
; CHECK-NEXT:    ret float [[ADD3]]
;
  %mul1 = fmul fast float %a, %c
  %mul2 = fmul fast float %a, %b
  %mul3 = fmul float %a, %b
  %mul4 = fmul fast float %a, %c
  %add1 = fadd fast  float %mul1, %mul3
  %add2 = fadd  fast float %mul4, %mul2
  %add3 = fadd fast float %add1, %add2
  ret float %add3
}
