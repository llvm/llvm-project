; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define <2 x double> @test_round_sd(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: @test_round_sd
; CHECK-NEXT: %1 = insertelement <2 x double> %a, double 1.000000e+00, i32 0
; CHECK-NEXT: %2 = insertelement <2 x double> %b, double 2.000000e+00, i32 1
; CHECK-NEXT: %3 = tail call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %1, <2 x double> %2, i32 10)
; CHECK-NEXT: ret <2 x double> %3
  %1 = insertelement <2 x double> %a, double 1.000000e+00, i32 0
  %2 = insertelement <2 x double> %b, double 2.000000e+00, i32 1
  %3 = tail call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %1, <2 x double> %2, i32 10)
  ret <2 x double> %3
}

define double @test_round_sd_0(double %a, double %b) {
; CHECK-LABEL: @test_round_sd_0
; CHECK-NEXT: %1 = insertelement <2 x double> undef, double %a, i32 0
; CHECK-NEXT: %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
; CHECK-NEXT: %3 = insertelement <2 x double> undef, double %b, i32 0
; CHECK-NEXT: %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
; CHECK-NEXT: %5 = tail call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %2, <2 x double> %4, i32 10)
; CHECK-NEXT: %6 = extractelement <2 x double> %5, i32 0
; CHECK-NEXT: ret double %6
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = insertelement <2 x double> undef, double %b, i32 0
  %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
  %5 = tail call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %2, <2 x double> %4, i32 10)
  %6 = extractelement <2 x double> %5, i32 0
  ret double %6 
}

define <4 x float> @test_round_ss(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: @test_round
; CHECK-NEXT: %1 = insertelement <4 x float> %a, float 1.000000e+00, i32 1
; CHECK-NEXT: %2 = insertelement <4 x float> %1, float 2.000000e+00, i32 2
; CHECK-NEXT: %3 = insertelement <4 x float> %2, float 3.000000e+00, i32 3
; CHECK-NEXT: %4 = insertelement <4 x float> %b, float 1.000000e+00, i32 1
; CHECK-NEXT: %5 = insertelement <4 x float> %4, float 2.000000e+00, i32 2
; CHECK-NEXT: %6 = insertelement <4 x float> %5, float 3.000000e+00, i32 3
; CHECK-NEXT: %7 = tail call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %3, <4 x float> %6, i32 10)
; CHECK-NEXT: ret <4 x float> %7
  %1 = insertelement <4 x float> %a, float 1.000000e+00, i32 1
  %2 = insertelement <4 x float> %1, float 2.000000e+00, i32 2
  %3 = insertelement <4 x float> %2, float 3.000000e+00, i32 3
  %4 = insertelement <4 x float> %b, float 1.000000e+00, i32 1
  %5 = insertelement <4 x float> %4, float 2.000000e+00, i32 2
  %6 = insertelement <4 x float> %5, float 3.000000e+00, i32 3
  %7 = tail call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %3, <4 x float> %6, i32 10)
  ret <4 x float> %7
}

define float @test_round_ss_0(float %a, float %b) {
; CHECK-LABEL: @test_round_ss_0
; CHECK-NEXT: %1 = insertelement <4 x float> undef, float %a, i32 0
; CHECK-NEXT: %2 = insertelement <4 x float> %1, float 1.000000e+00, i32 1
; CHECK-NEXT: %3 = insertelement <4 x float> %2, float 2.000000e+00, i32 2
; CHECK-NEXT: %4 = insertelement <4 x float> %3, float 3.000000e+00, i32 3
; CHECK-NEXT: %5 = insertelement <4 x float> undef, float %b, i32 0
; CHECK-NEXT: %6 = insertelement <4 x float> %5, float 4.000000e+00, i32 1
; CHECK-NEXT: %7 = insertelement <4 x float> %6, float 5.000000e+00, i32 2
; CHECK-NEXT: %8 = insertelement <4 x float> %7, float 6.000000e+00, i32 3
; CHECK-NEXT: %9 = tail call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %4, <4 x float> %8, i32 10)
; CHECK-NEXT: %r = extractelement <4 x float> %9, i32 0
; CHECK-NEXT: ret float %r
  %1 = insertelement <4 x float> undef, float %a, i32 0
  %2 = insertelement <4 x float> %1, float 1.000000e+00, i32 1
  %3 = insertelement <4 x float> %2, float 2.000000e+00, i32 2
  %4 = insertelement <4 x float> %3, float 3.000000e+00, i32 3
  %5 = insertelement <4 x float> undef, float %b, i32 0
  %6 = insertelement <4 x float> %5, float 4.000000e+00, i32 1
  %7 = insertelement <4 x float> %6, float 5.000000e+00, i32 2
  %8 = insertelement <4 x float> %7, float 6.000000e+00, i32 3
  %9 = tail call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %4, <4 x float> %8, i32 10)
  %r = extractelement <4 x float> %9, i32 0
  ret float %r
}

declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone
declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone