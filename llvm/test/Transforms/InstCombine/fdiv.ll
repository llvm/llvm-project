; RUN: opt -S -instcombine < %s | FileCheck %s

define float @test1(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x3810000000000000
  ret float %div

; CHECK: @test1
; CHECK-NEXT: fmul float %x, 0x47D0000000000000
}

define float @test2(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x47E0000000000000
  ret float %div

; CHECK: @test2
; CHECK-NEXT: fdiv float %x, 0x47E0000000000000
}

define float @test3(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x36A0000000000000
  ret float %div

; CHECK: @test3
; CHECK-NEXT: fdiv float %x, 0x36A0000000000000
}
