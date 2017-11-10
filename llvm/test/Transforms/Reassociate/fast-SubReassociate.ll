; RUN: opt < %s -reassociate -constprop -instcombine -S | FileCheck %s

define float @test1(float %A, float %B) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[W:%.*]] = fadd float %B, 5.000000e+00
; CHECK-NEXT:    [[X:%.*]] = fadd float %A, -7.000000e+00
; CHECK-NEXT:    [[Y:%.*]] = fsub float [[X]], [[W]]
; CHECK-NEXT:    [[Z:%.*]] = fadd float [[Y]], 1.200000e+01
; CHECK-NEXT:    ret float [[Z]]
;
  %W = fadd float 5.0, %B
  %X = fadd float -7.0, %A
  %Y = fsub float %X, %W
  %Z = fadd float %Y, 12.0
  ret float %Z
}

; With sub reassociation, constant folding can eliminate all of the constants.
define float @test2(float %A, float %B) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[Z:%.*]] = fsub fast float %A, %B
; CHECK-NEXT:    ret float [[Z]]
;
  %W = fadd fast float %B, 5.000000e+00
  %X = fadd fast float %A, -7.000000e+00
  %Y = fsub fast float %X, %W
  %Z = fadd fast float %Y, 1.200000e+01
  ret float %Z

}

define float @test3(float %A, float %B, float %C, float %D) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[M:%.*]] = fadd float %A, 1.200000e+01
; CHECK-NEXT:    [[N:%.*]] = fadd float [[M]], %B
; CHECK-NEXT:    [[O:%.*]] = fadd float [[N]], %C
; CHECK-NEXT:    [[P:%.*]] = fsub float %D, [[O]]
; CHECK-NEXT:    [[Q:%.*]] = fadd float [[P]], 1.200000e+01
; CHECK-NEXT:    ret float [[Q]]
;
  %M = fadd float %A, 1.200000e+01
  %N = fadd float %M, %B
  %O = fadd float %N, %C
  %P = fsub float %D, %O
  %Q = fadd float %P, 1.200000e+01
  ret float %Q
}

; With sub reassociation, constant folding can eliminate the two 12 constants.
define float @test4(float %A, float %B, float %C, float %D) {
; FIXME: InstCombine should be able to get us to the following:
; %sum = fadd fast float %B, %A
; %sum1 = fadd fast float %sum, %C
; %Q = fsub fast float %D, %sum1
; ret i32 %Q
; CHECK-LABEL: @test4(
; CHECK-NEXT:    [[B_NEG:%.*]] = fsub fast float -0.000000e+00, %B
; CHECK-NEXT:    [[O_NEG:%.*]] = fsub fast float [[B_NEG]], %A
; CHECK-NEXT:    [[P:%.*]] = fsub fast float [[O_NEG]], %C
; CHECK-NEXT:    [[Q:%.*]] = fadd fast float [[P]], %D
; CHECK-NEXT:    ret float [[Q]]
;
  %M = fadd fast float 1.200000e+01, %A
  %N = fadd fast float %M, %B
  %O = fadd fast float %N, %C
  %P = fsub fast float %D, %O
  %Q = fadd fast float 1.200000e+01, %P
  ret float %Q
}

