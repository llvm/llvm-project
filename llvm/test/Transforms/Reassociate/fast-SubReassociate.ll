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

; Check again using minimal subset of FMF.
; Both 'reassoc' and 'nsz' are required.
define float @test2_minimal(float %A, float %B) {
; CHECK-LABEL: @test2_minimal(
; CHECK-NEXT:    [[Z:%.*]] = fsub reassoc nsz float %A, %B
; CHECK-NEXT:    ret float [[Z]]
;
  %W = fadd reassoc nsz float %B, 5.000000e+00
  %X = fadd reassoc nsz float %A, -7.000000e+00
  %Y = fsub reassoc nsz float %X, %W
  %Z = fadd reassoc nsz float %Y, 1.200000e+01
  ret float %Z
}

; Verify the fold is not done with only 'reassoc' ('nsz' is required).
define float @test2_reassoc(float %A, float %B) {
; CHECK-LABEL: @test2_reassoc(
; CHECK-NEXT:    [[W:%.*]] = fadd reassoc float %B, 5.000000e+00
; CHECK-NEXT:    [[X:%.*]] = fadd reassoc float %A, -7.000000e+00
; CHECK-NEXT:    [[Y:%.*]] = fsub reassoc float [[X]], [[W]]
; CHECK-NEXT:    [[Z:%.*]] = fadd reassoc float [[Y]], 1.200000e+01
; CHECK-NEXT:    ret float [[Z]]
;
  %W = fadd reassoc float %B, 5.000000e+00
  %X = fadd reassoc float %A, -7.000000e+00
  %Y = fsub reassoc float %X, %W
  %Z = fadd reassoc float %Y, 1.200000e+01
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

; Check again using minimal subset of FMF.

define float @test4_reassoc(float %A, float %B, float %C, float %D) {
; CHECK-LABEL: @test4_reassoc(
; CHECK-NEXT:    [[M:%.*]] = fadd reassoc float %A, 1.200000e+01
; CHECK-NEXT:    [[N:%.*]] = fadd reassoc float [[M]], %B
; CHECK-NEXT:    [[O:%.*]] = fadd reassoc float [[N]], %C
; CHECK-NEXT:    [[P:%.*]] = fsub reassoc float %D, [[O]]
; CHECK-NEXT:    [[Q:%.*]] = fadd reassoc float [[P]], 1.200000e+01
; CHECK-NEXT:    ret float [[Q]]
;
  %M = fadd reassoc float 1.200000e+01, %A
  %N = fadd reassoc float %M, %B
  %O = fadd reassoc float %N, %C
  %P = fsub reassoc float %D, %O
  %Q = fadd reassoc float 1.200000e+01, %P
  ret float %Q
}

