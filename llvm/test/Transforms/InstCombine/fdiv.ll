; RUN: opt -S -instcombine < %s | FileCheck %s

define float @test1(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x3810000000000000
  ret float %div

; CHECK-LABEL: @test1(
; CHECK-NEXT: fmul float %x, 0x47D0000000000000
}

define float @test2(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x47E0000000000000
  ret float %div

; CHECK-LABEL: @test2(
; CHECK-NEXT: fdiv float %x, 0x47E0000000000000
}

define float @test3(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x36A0000000000000
  ret float %div

; CHECK-LABEL: @test3(
; CHECK-NEXT: fdiv float %x, 0x36A0000000000000
}

define float @test4(float %x) nounwind readnone ssp {
  %div = fdiv fast float %x, 8.0
  ret float %div

; CHECK-LABEL: @test4(
; CHECK-NEXT: fmul fast float %x, 1.250000e-01
}

define float @test5(float %x, float %y, float %z) nounwind readnone ssp {
  %div1 = fdiv fast float %x, %y
  %div2 = fdiv fast float %div1, %z
  ret float %div2
; CHECK-LABEL: @test5(
; CHECK-NEXT: fmul fast
; CHECK-NEXT: fdiv fast
}

define float @test6(float %x, float %y, float %z) nounwind readnone ssp {
  %div1 = fdiv fast float %x, %y
  %div2 = fdiv fast float %z, %div1
  ret float %div2
; CHECK-LABEL: @test6(
; CHECK-NEXT: fmul fast
; CHECK-NEXT: fdiv fast
}

; CHECK-LABEL @fdiv_fneg_fneg(
; CHECK: %div = fdiv float %x, %y
define float @fdiv_fneg_fneg(float %x, float %y) {
  %x.fneg = fsub float -0.0, %x
  %y.fneg = fsub float -0.0, %y
  %div = fdiv float %x.fneg, %y.fneg
  ret float %div
}

; CHECK-LABEL @fdiv_fneg_fneg_fast(
; CHECK: %div = fdiv fast float %x, %y
define float @fdiv_fneg_fneg_fast(float %x, float %y) {
  %x.fneg = fsub float -0.0, %x
  %y.fneg = fsub float -0.0, %y
  %div = fdiv fast float %x.fneg, %y.fneg
  ret float %div
}
