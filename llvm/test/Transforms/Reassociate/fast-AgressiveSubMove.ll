; RUN: opt < %s -reassociate -S | FileCheck %s

define float @test1(float %A) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[X:%.*]] = fadd float %A, 1.000000e+00
; CHECK-NEXT:    [[Y:%.*]] = fadd float %A, 1.000000e+00
; CHECK-NEXT:    [[R:%.*]] = fsub float [[X]], [[Y]]
; CHECK-NEXT:    ret float [[R]]
;
  %X = fadd float %A, 1.000000e+00
  %Y = fadd float %A, 1.000000e+00
  %r = fsub float %X, %Y
  ret float %r
}

define float @test2(float %A) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    ret float 0.000000e+00
;
  %X = fadd fast float 1.000000e+00, %A
  %Y = fadd fast float 1.000000e+00, %A
  %r = fsub fast float %X, %Y
  ret float %r
}

; Check again using minimal subset of FMF.

define float @test2_reassoc(float %A) {
; CHECK-LABEL: @test2_reassoc(
; CHECK-NEXT:    [[X:%.*]] = fadd reassoc float %A, 1.000000e+00
; CHECK-NEXT:    [[Y:%.*]] = fadd reassoc float %A, 1.000000e+00
; CHECK-NEXT:    [[R:%.*]] = fsub reassoc float [[X]], [[Y]]
; CHECK-NEXT:    ret float [[R]]
;
  %X = fadd reassoc float 1.000000e+00, %A
  %Y = fadd reassoc float 1.000000e+00, %A
  %r = fsub reassoc float %X, %Y
  ret float %r
}

