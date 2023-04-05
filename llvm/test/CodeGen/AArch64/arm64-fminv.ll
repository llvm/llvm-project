; RUN: llc -mtriple=arm64-linux-gnu -o - %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=arm64-linux-gnu -o - %s | FileCheck %s

define float @test_fminv_v2f32(<2 x float> %in) {
; CHECK: test_fminv_v2f32:
; CHECK: fminp s0, v0.2s
  %min = call float @llvm.aarch64.neon.fminv.f32.v2f32(<2 x float> %in)
  ret float %min
}

define float @test_fminv_v4f32(<4 x float> %in) {
; CHECK: test_fminv_v4f32:
; CHECK: fminv s0, v0.4s
  %min = call float @llvm.aarch64.neon.fminv.f32.v4f32(<4 x float> %in)
  ret float %min
}

define double @test_fminv_v2f64(<2 x double> %in) {
; CHECK: test_fminv_v2f64:
; CHECK: fminp d0, v0.2d
  %min = call double @llvm.aarch64.neon.fminv.f64.v2f64(<2 x double> %in)
  ret double %min
}

declare float @llvm.aarch64.neon.fminv.f32.v2f32(<2 x float>)
declare float @llvm.aarch64.neon.fminv.f32.v4f32(<4 x float>)
declare double @llvm.aarch64.neon.fminv.f64.v2f64(<2 x double>)

define float @test_fmaxv_v2f32(<2 x float> %in) {
; CHECK: test_fmaxv_v2f32:
; CHECK: fmaxp s0, v0.2s
  %max = call float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float> %in)
  ret float %max
}

define float @test_fmaxv_v4f32(<4 x float> %in) {
; CHECK: test_fmaxv_v4f32:
; CHECK: fmaxv s0, v0.4s
  %max = call float @llvm.aarch64.neon.fmaxv.f32.v4f32(<4 x float> %in)
  ret float %max
}

define double @test_fmaxv_v2f64(<2 x double> %in) {
; CHECK: test_fmaxv_v2f64:
; CHECK: fmaxp d0, v0.2d
  %max = call double @llvm.aarch64.neon.fmaxv.f64.v2f64(<2 x double> %in)
  ret double %max
}

declare float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float>)
declare float @llvm.aarch64.neon.fmaxv.f32.v4f32(<4 x float>)
declare double @llvm.aarch64.neon.fmaxv.f64.v2f64(<2 x double>)

define float @test_fminnmv_v2f32(<2 x float> %in) {
; CHECK: test_fminnmv_v2f32:
; CHECK: fminnmp s0, v0.2s
  %minnm = call float @llvm.aarch64.neon.fminnmv.f32.v2f32(<2 x float> %in)
  ret float %minnm
}

define float @test_fminnmv_v4f32(<4 x float> %in) {
; CHECK: test_fminnmv_v4f32:
; CHECK: fminnmv s0, v0.4s
  %minnm = call float @llvm.aarch64.neon.fminnmv.f32.v4f32(<4 x float> %in)
  ret float %minnm
}

define double @test_fminnmv_v2f64(<2 x double> %in) {
; CHECK: test_fminnmv_v2f64:
; CHECK: fminnmp d0, v0.2d
  %minnm = call double @llvm.aarch64.neon.fminnmv.f64.v2f64(<2 x double> %in)
  ret double %minnm
}

declare float @llvm.aarch64.neon.fminnmv.f32.v2f32(<2 x float>)
declare float @llvm.aarch64.neon.fminnmv.f32.v4f32(<4 x float>)
declare double @llvm.aarch64.neon.fminnmv.f64.v2f64(<2 x double>)

define float @test_fmaxnmv_v2f32(<2 x float> %in) {
; CHECK: test_fmaxnmv_v2f32:
; CHECK: fmaxnmp s0, v0.2s
  %maxnm = call float @llvm.aarch64.neon.fmaxnmv.f32.v2f32(<2 x float> %in)
  ret float %maxnm
}

define float @test_fmaxnmv_v4f32(<4 x float> %in) {
; CHECK: test_fmaxnmv_v4f32:
; CHECK: fmaxnmv s0, v0.4s
  %maxnm = call float @llvm.aarch64.neon.fmaxnmv.f32.v4f32(<4 x float> %in)
  ret float %maxnm
}

define double @test_fmaxnmv_v2f64(<2 x double> %in) {
; CHECK: test_fmaxnmv_v2f64:
; CHECK: fmaxnmp d0, v0.2d
  %maxnm = call double @llvm.aarch64.neon.fmaxnmv.f64.v2f64(<2 x double> %in)
  ret double %maxnm
}

declare float @llvm.aarch64.neon.fmaxnmv.f32.v2f32(<2 x float>)
declare float @llvm.aarch64.neon.fmaxnmv.f32.v4f32(<4 x float>)
declare double @llvm.aarch64.neon.fmaxnmv.f64.v2f64(<2 x double>)
