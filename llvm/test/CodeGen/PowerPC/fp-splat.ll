; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefix=CHECK-P8 -check-prefix=CHECK
; RUN: llc -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s -check-prefix=CHECK-P7 -check-prefix=CHECK

define <4 x float> @test1(float %a) {
entry:
; CHECK-LABEL: test1
  %vecins = insertelement <4 x float> undef, float %a, i32 0
  %vecins1 = insertelement <4 x float> %vecins, float %a, i32 1
  %vecins2 = insertelement <4 x float> %vecins1, float %a, i32 2
  %vecins3 = insertelement <4 x float> %vecins2, float %a, i32 3
  ret <4 x float> %vecins3
; CHECK-P8: xscvdpspn
; CHECK-P7-NOT: xscvdpspn
; CHECK: blr
}

define <2 x double> @test2(double %a) {
entry:
; CHECK-LABEL: test2
  %vecins = insertelement <2 x double> undef, double %a, i32 0
  %vecins1 = insertelement <2 x double> %vecins, double %a, i32 1
  ret <2 x double> %vecins1
; CHECK-P8: xxspltd
; CHECK-P7: xxspltd
; CHECK: blr
}

