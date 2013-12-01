; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define float @test_vabds_f32(float %a, float %b) {
; CHECK-LABEL: test_vabds_f32
; CHECK: fabd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vabd.i = insertelement <1 x float> undef, float %a, i32 0
  %vabd1.i = insertelement <1 x float> undef, float %b, i32 0
  %vabd2.i = call <1 x float> @llvm.aarch64.neon.vabd.v1f32(<1 x float> %vabd.i, <1 x float> %vabd1.i)
  %0 = extractelement <1 x float> %vabd2.i, i32 0
  ret float %0
}

define double @test_vabdd_f64(double %a, double %b) {
; CHECK-LABEL: test_vabdd_f64
; CHECK: fabd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vabd.i = insertelement <1 x double> undef, double %a, i32 0
  %vabd1.i = insertelement <1 x double> undef, double %b, i32 0
  %vabd2.i = call <1 x double> @llvm.aarch64.neon.vabd.v1f64(<1 x double> %vabd.i, <1 x double> %vabd1.i)
  %0 = extractelement <1 x double> %vabd2.i, i32 0
  ret double %0
}

declare <1 x double> @llvm.aarch64.neon.vabd.v1f64(<1 x double>, <1 x double>)
declare <1 x float> @llvm.aarch64.neon.vabd.v1f32(<1 x float>, <1 x float>)
