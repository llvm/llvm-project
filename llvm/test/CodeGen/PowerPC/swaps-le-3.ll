; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -O3 < %s | FileCheck %s

; This test verifies that VSX swap optimization works for the
; doubleword splat idiom.

@a = external global <2 x double>, align 16
@b = external global <2 x double>, align 16

define void @test(double %s) {
entry:
  %0 = insertelement <2 x double> undef, double %s, i32 0
  %1 = shufflevector <2 x double> %0, <2 x double> undef, <2 x i32> zeroinitializer
  %2 = load <2 x double>, ptr @a, align 16
  %3 = fadd <2 x double> %0, %2
  store <2 x double> %3, ptr @b, align 16
  ret void
}

; CHECK-LABEL: @test
; CHECK-DAG: xxspltd
; CHECK-DAG: lxvd2x
; CHECK: xvadddp
; CHECK: stxvd2x
; CHECK-NOT: xxswapd
