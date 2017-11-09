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

