; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

;; Scalar Floating-point Convert

define float @test_vcvtxn(double %a) {
; CHECK: test_vcvtxn
; CHECK: fcvtxn {{s[0-9]}}, {{d[0-9]}}
entry:
  %vcvtf.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtf1.i = tail call <1 x float> @llvm.aarch64.neon.fcvtxn.v1f32.v1f64(<1 x double> %vcvtf.i)
  %0 = extractelement <1 x float> %vcvtf1.i, i32 0
  ret float %0
}

declare <1 x float> @llvm.aarch64.neon.fcvtxn.v1f32.v1f64(<1 x double>)

define i32 @test_vcvtass(float %a) {
; CHECK: test_vcvtass
; CHECK: fcvtas {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtas.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtas1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtas.v1i32.v1f32(<1 x float> %vcvtas.i)
  %0 = extractelement <1 x i32> %vcvtas1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtas.v1i32.v1f32(<1 x float>)

define i64 @test_test_vcvtasd(double %a) {
; CHECK: test_test_vcvtasd
; CHECK: fcvtas {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtas.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtas1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtas.v1i64.v1f64(<1 x double> %vcvtas.i)
  %0 = extractelement <1 x i64> %vcvtas1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtas.v1i64.v1f64(<1 x double>)

define i32 @test_vcvtaus(float %a) {
; CHECK: test_vcvtaus
; CHECK: fcvtau {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtau.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtau1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtau.v1i32.v1f32(<1 x float> %vcvtau.i)
  %0 = extractelement <1 x i32> %vcvtau1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtau.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtaud(double %a) {
; CHECK: test_vcvtaud
; CHECK: fcvtau {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtau.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtau1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtau.v1i64.v1f64(<1 x double> %vcvtau.i)
  %0 = extractelement <1 x i64> %vcvtau1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtau.v1i64.v1f64(<1 x double>) 

define i32 @test_vcvtmss(float %a) {
; CHECK: test_vcvtmss
; CHECK: fcvtms {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtms.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtms1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtms.v1i32.v1f32(<1 x float> %vcvtms.i)
  %0 = extractelement <1 x i32> %vcvtms1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtms.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtmd_s64_f64(double %a) {
; CHECK: test_vcvtmd_s64_f64
; CHECK: fcvtms {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtms.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtms1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtms.v1i64.v1f64(<1 x double> %vcvtms.i)
  %0 = extractelement <1 x i64> %vcvtms1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtms.v1i64.v1f64(<1 x double>)

define i32 @test_vcvtmus(float %a) {
; CHECK: test_vcvtmus
; CHECK: fcvtmu {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtmu.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtmu1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtmu.v1i32.v1f32(<1 x float> %vcvtmu.i)
  %0 = extractelement <1 x i32> %vcvtmu1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtmu.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtmud(double %a) {
; CHECK: test_vcvtmud
; CHECK: fcvtmu {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtmu.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtmu1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtmu.v1i64.v1f64(<1 x double> %vcvtmu.i)
  %0 = extractelement <1 x i64> %vcvtmu1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtmu.v1i64.v1f64(<1 x double>)

define i32 @test_vcvtnss(float %a) {
; CHECK: test_vcvtnss
; CHECK: fcvtns {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtns.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtns1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtns.v1i32.v1f32(<1 x float> %vcvtns.i)
  %0 = extractelement <1 x i32> %vcvtns1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtns.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtnd_s64_f64(double %a) {
; CHECK: test_vcvtnd_s64_f64
; CHECK: fcvtns {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtns.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtns1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtns.v1i64.v1f64(<1 x double> %vcvtns.i)
  %0 = extractelement <1 x i64> %vcvtns1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtns.v1i64.v1f64(<1 x double>)

define i32 @test_vcvtnus(float %a) {
; CHECK: test_vcvtnus
; CHECK: fcvtnu {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtnu.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtnu1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtnu.v1i32.v1f32(<1 x float> %vcvtnu.i)
  %0 = extractelement <1 x i32> %vcvtnu1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtnu.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtnud(double %a) {
; CHECK: test_vcvtnud
; CHECK: fcvtnu {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtnu.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtnu1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtnu.v1i64.v1f64(<1 x double> %vcvtnu.i)
  %0 = extractelement <1 x i64> %vcvtnu1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtnu.v1i64.v1f64(<1 x double>)

define i32 @test_vcvtpss(float %a) {
; CHECK: test_vcvtpss
; CHECK: fcvtps {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtps.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtps1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtps.v1i32.v1f32(<1 x float> %vcvtps.i)
  %0 = extractelement <1 x i32> %vcvtps1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtps.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtpd_s64_f64(double %a) {
; CHECK: test_vcvtpd_s64_f64
; CHECK: fcvtps {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtps.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtps1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtps.v1i64.v1f64(<1 x double> %vcvtps.i)
  %0 = extractelement <1 x i64> %vcvtps1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtps.v1i64.v1f64(<1 x double>)

define i32 @test_vcvtpus(float %a) {
; CHECK: test_vcvtpus
; CHECK: fcvtpu {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtpu.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtpu1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtpu.v1i32.v1f32(<1 x float> %vcvtpu.i)
  %0 = extractelement <1 x i32> %vcvtpu1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtpu.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtpud(double %a) {
; CHECK: test_vcvtpud
; CHECK: fcvtpu {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtpu.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtpu1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtpu.v1i64.v1f64(<1 x double> %vcvtpu.i)
  %0 = extractelement <1 x i64> %vcvtpu1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtpu.v1i64.v1f64(<1 x double>)

define i32 @test_vcvtss(float %a) {
; CHECK: test_vcvtss
; CHECK: fcvtzs {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtzs.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtzs1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtzs.v1i32.v1f32(<1 x float> %vcvtzs.i)
  %0 = extractelement <1 x i32> %vcvtzs1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtzs.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtd_s64_f64(double %a) {
; CHECK: test_vcvtd_s64_f64
; CHECK: fcvtzs {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvzs.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvzs1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtzs.v1i64.v1f64(<1 x double> %vcvzs.i)
  %0 = extractelement <1 x i64> %vcvzs1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtzs.v1i64.v1f64(<1 x double>)

define i32 @test_vcvtus(float %a) {
; CHECK: test_vcvtus
; CHECK: fcvtzu {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcvtzu.i = insertelement <1 x float> undef, float %a, i32 0
  %vcvtzu1.i = tail call <1 x i32> @llvm.aarch64.neon.fcvtzu.v1i32.v1f32(<1 x float> %vcvtzu.i)
  %0 = extractelement <1 x i32> %vcvtzu1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.fcvtzu.v1i32.v1f32(<1 x float>)

define i64 @test_vcvtud(double %a) {
; CHECK: test_vcvtud
; CHECK: fcvtzu {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcvtzu.i = insertelement <1 x double> undef, double %a, i32 0
  %vcvtzu1.i = tail call <1 x i64> @llvm.aarch64.neon.fcvtzu.v1i64.v1f64(<1 x double> %vcvtzu.i)
  %0 = extractelement <1 x i64> %vcvtzu1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.fcvtzu.v1i64.v1f64(<1 x double>)
