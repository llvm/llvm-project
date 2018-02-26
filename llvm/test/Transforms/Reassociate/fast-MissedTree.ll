; RUN: opt < %s -reassociate -instcombine -S | FileCheck %s

define float @test1(float %A, float %B) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[Z:%.*]] = fadd fast float %A, %B
; CHECK-NEXT:    ret float [[Z]]
;
  %W = fadd fast float %B, -5.0
  %Y = fadd fast float %A, 5.0
  %Z = fadd fast float %W, %Y
  ret float %Z
}

; Check again using minimal subset of FMF.

define float @test1_reassoc(float %A, float %B) {
; CHECK-LABEL: @test1_reassoc(
; CHECK-NEXT:    [[W:%.*]] = fadd reassoc float %B, -5.000000e+00
; CHECK-NEXT:    [[Y:%.*]] = fadd reassoc float %A, 5.000000e+00
; CHECK-NEXT:    [[Z:%.*]] = fadd reassoc float [[Y]], [[W]]
; CHECK-NEXT:    ret float [[Z]]
;
  %W = fadd reassoc float %B, -5.0
  %Y = fadd reassoc float %A, 5.0
  %Z = fadd reassoc float %W, %Y
  ret float %Z
}

