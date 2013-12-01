; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

define float @test_vcvts_f32_s32(i32 %a) {
; CHECK: test_vcvts_f32_s32
; CHECK: scvtf {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %0 = call float @llvm.aarch64.neon.vcvtf32.s32(<1 x i32> %vcvtf.i)
  ret float %0
}

declare float @llvm.aarch64.neon.vcvtf32.s32(<1 x i32>)

define double @test_vcvtd_f64_s64(i64 %a) {
; CHECK: test_vcvtd_f64_s64
; CHECK: scvtf {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %0 = call double @llvm.aarch64.neon.vcvtf64.s64(<1 x i64> %vcvtf.i)
  ret double %0
}

declare double @llvm.aarch64.neon.vcvtf64.s64(<1 x i64>)

define float @test_vcvts_f32_u32(i32 %a) {
; CHECK: test_vcvts_f32_u32
; CHECK: ucvtf {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %0 = call float @llvm.aarch64.neon.vcvtf32.u32(<1 x i32> %vcvtf.i)
  ret float %0
}

declare float @llvm.aarch64.neon.vcvtf32.u32(<1 x i32>)

define double @test_vcvtd_f64_u64(i64 %a) {
; CHECK: test_vcvtd_f64_u64
; CHECK: ucvtf {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %0 = call double @llvm.aarch64.neon.vcvtf64.u64(<1 x i64> %vcvtf.i)
  ret double %0
}

declare double @llvm.aarch64.neon.vcvtf64.u64(<1 x i64>)

define float @test_vcvts_n_f32_s32(i32 %a) {
; CHECK: test_vcvts_n_f32_s32
; CHECK: scvtf {{s[0-9]+}}, {{s[0-9]+}}, #1
entry:
  %vcvtf = insertelement <1 x i32> undef, i32 %a, i32 0
  %0 = call float @llvm.aarch64.neon.vcvtf32.n.s32(<1 x i32> %vcvtf, i32 1)
  ret float %0
}

declare float @llvm.aarch64.neon.vcvtf32.n.s32(<1 x i32>, i32)

define double @test_vcvtd_n_f64_s64(i64 %a) {
; CHECK: test_vcvtd_n_f64_s64
; CHECK: scvtf {{d[0-9]+}}, {{d[0-9]+}}, #1
entry:
  %vcvtf = insertelement <1 x i64> undef, i64 %a, i32 0
  %0 = call double @llvm.aarch64.neon.vcvtf64.n.s64(<1 x i64> %vcvtf, i32 1)
  ret double %0
}

declare double @llvm.aarch64.neon.vcvtf64.n.s64(<1 x i64>, i32)

define float @test_vcvts_n_f32_u32(i32 %a) {
; CHECK: test_vcvts_n_f32_u32
; CHECK: ucvtf {{s[0-9]+}}, {{s[0-9]+}}, #1
entry:
  %vcvtf = insertelement <1 x i32> undef, i32 %a, i32 0
  %0 = call float @llvm.aarch64.neon.vcvtf32.n.u32(<1 x i32> %vcvtf, i32 1)
  ret float %0
}

declare float @llvm.aarch64.neon.vcvtf32.n.u32(<1 x i32>, i32)

define double @test_vcvtd_n_f64_u64(i64 %a) {
; CHECK: test_vcvtd_n_f64_u64
; CHECK: ucvtf {{d[0-9]+}}, {{d[0-9]+}}, #1
entry:
  %vcvtf = insertelement <1 x i64> undef, i64 %a, i32 0
  %0 = call double @llvm.aarch64.neon.vcvtf64.n.u64(<1 x i64> %vcvtf, i32 1)
  ret double %0
}

declare double @llvm.aarch64.neon.vcvtf64.n.u64(<1 x i64>, i32)

define i32 @test_vcvts_n_s32_f32(float %a) {
; CHECK: test_vcvts_n_s32_f32
; CHECK: fcvtzs {{s[0-9]+}}, {{s[0-9]+}}, #1
entry:
  %fcvtzs = insertelement <1 x float> undef, float %a, i32 0
  %fcvtzs1 = call <1 x i32> @llvm.aarch64.neon.vcvts.n.s32.f32(<1 x float> %fcvtzs, i32 1)
  %0 = extractelement <1 x i32> %fcvtzs1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vcvts.n.s32.f32(<1 x float>, i32)

define i64 @test_vcvtd_n_s64_f64(double %a) {
; CHECK: test_vcvtd_n_s64_f64
; CHECK: fcvtzs {{d[0-9]+}}, {{d[0-9]+}}, #1
entry:
  %fcvtzs = insertelement <1 x double> undef, double %a, i32 0
  %fcvtzs1 = call <1 x i64> @llvm.aarch64.neon.vcvtd.n.s64.f64(<1 x double> %fcvtzs, i32 1)
  %0 = extractelement <1 x i64> %fcvtzs1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vcvtd.n.s64.f64(<1 x double>, i32)

define i32 @test_vcvts_n_u32_f32(float %a) {
; CHECK: test_vcvts_n_u32_f32
; CHECK: fcvtzu {{s[0-9]+}}, {{s[0-9]+}}, #32
entry:
  %fcvtzu = insertelement <1 x float> undef, float %a, i32 0
  %fcvtzu1 = call <1 x i32> @llvm.aarch64.neon.vcvts.n.u32.f32(<1 x float> %fcvtzu, i32 32)
  %0 = extractelement <1 x i32> %fcvtzu1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vcvts.n.u32.f32(<1 x float>, i32)

define i64 @test_vcvtd_n_u64_f64(double %a) {
; CHECK: test_vcvtd_n_u64_f64
; CHECK: fcvtzu {{d[0-9]+}}, {{d[0-9]+}}, #64
entry:
  %fcvtzu = insertelement <1 x double> undef, double %a, i32 0
  %fcvtzu1 = tail call <1 x i64> @llvm.aarch64.neon.vcvtd.n.u64.f64(<1 x double> %fcvtzu, i32 64)
  %0 = extractelement <1 x i64> %fcvtzu1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vcvtd.n.u64.f64(<1 x double>, i32)
