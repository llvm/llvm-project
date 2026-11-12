; RUN: llc -mtriple=armv6-apple-ios -mcpu=arm1136jf-s -o - %s | FileCheck %s --check-prefix=CHECK-HARD
; RUN: llc -mtriple=thumbv6-apple-ios -mcpu=arm1136jf-s -o - %s | FileCheck %s --check-prefix=CHECK-SOFTISH
; RUN: llc -mtriple=armv7s-apple-ios -mattr=+soft-float -mcpu=arm1136jf-s -o - %s | FileCheck %s --check-prefix=CHECK-SOFT

define float @test_call(float %a, float %b) {
; CHECK-HARD: vadd.f32 {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-SOFTISH: bl ___addsf3vfp
; CHECK-SOFT: bl ___addsf3{{$}}
  %sum = fadd float %a, %b
  ret float %sum
}
