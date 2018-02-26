; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s
; RUN: llc < %s -O0 -fast-isel -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <2 x double> @test_vcvt_f64_f32(<2 x float> %x) nounwind readnone ssp {
; CHECK-LABEL: test_vcvt_f64_f32:
  %vcvt1.i = fpext <2 x float> %x to <2 x double>
; CHECK: fcvtl	v0.2d, v0.2s
  ret <2 x double> %vcvt1.i
; CHECK: ret
}

define <2 x double> @test_vcvt_high_f64_f32(<4 x float> %x) nounwind readnone ssp {
; CHECK-LABEL: test_vcvt_high_f64_f32:
  %cvt_in = shufflevector <4 x float> %x, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  %vcvt1.i = fpext <2 x float> %cvt_in to <2 x double>
; CHECK: fcvtl2	v0.2d, v0.4s
  ret <2 x double> %vcvt1.i
; CHECK: ret
}

define <2 x float> @test_vcvt_f32_f64(<2 x double> %v) nounwind readnone ssp {
; CHECK-LABEL: test_vcvt_f32_f64:
  %vcvt1.i = fptrunc <2 x double> %v to <2 x float>
; CHECK: fcvtn
  ret <2 x float> %vcvt1.i
; CHECK: ret
}

define <4 x float> @test_vcvt_high_f32_f64(<2 x float> %x, <2 x double> %v) nounwind readnone ssp {
; CHECK-LABEL: test_vcvt_high_f32_f64:

  %cvt = fptrunc <2 x double> %v to <2 x float>
  %vcvt2.i = shufflevector <2 x float> %x, <2 x float> %cvt, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK: fcvtn2
  ret <4 x float> %vcvt2.i
; CHECK: ret
}

define <2 x float> @test_vcvtx_f32_f64(<2 x double> %v) nounwind readnone ssp {
; CHECK-LABEL: test_vcvtx_f32_f64:
  %vcvtx1.i = tail call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double> %v) nounwind
; CHECK: fcvtxn
  ret <2 x float> %vcvtx1.i
; CHECK: ret
}

define <4 x float> @test_vcvtx_high_f32_f64(<2 x float> %x, <2 x double> %v) nounwind readnone ssp {
; CHECK-LABEL: test_vcvtx_high_f32_f64:
  %vcvtx2.i = tail call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double> %v) nounwind
  %res = shufflevector <2 x float> %x, <2 x float> %vcvtx2.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK: fcvtxn2
  ret <4 x float> %res
; CHECK: ret
}


declare <2 x double> @llvm.aarch64.neon.vcvthighfp2df(<4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.vcvtfp2df(<2 x float>) nounwind readnone

declare <2 x float> @llvm.aarch64.neon.vcvtdf2fp(<2 x double>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.vcvthighdf2fp(<2 x float>, <2 x double>) nounwind readnone

declare <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double>) nounwind readnone

define i16 @to_half(float %in) {
; CHECK-LABEL: to_half:
; CHECK: fcvt h[[HALFVAL:[0-9]+]], s0
; CHECK: fmov {{w[0-9]+}}, {{s[0-9]+}}
  %res = call i16 @llvm.convert.to.fp16.f32(float %in)
  ret i16 %res
}

define float @from_half(i16 %in) {
; CHECK-LABEL: from_half:
; CHECK: fmov {{s[0-9]+}}, {{w[0-9]+}}
; CHECK: fcvt s0, {{h[0-9]+}}
  %res = call float @llvm.convert.from.fp16.f32(i16 %in)
  ret float %res
}

declare float @llvm.convert.from.fp16.f32(i16) #1
declare i16 @llvm.convert.to.fp16.f32(float) #1
