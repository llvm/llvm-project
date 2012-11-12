; RUN: llc < %s -march=x86 -mattr=+avx,+f16c | FileCheck %s

define <4 x float> @test_x86_vcvtph2ps_128(<8 x i16> %a0) {
  ; CHECK: vcvtph2ps
  %res = call <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16>) nounwind readonly


define <8 x float> @test_x86_vcvtph2ps_256(<8 x i16> %a0) {
  ; CHECK: vcvtph2ps
  %res = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readonly


define <8 x i16> @test_x86_vcvtps2ph_128(<4 x float> %a0) {
  ; CHECK: vcvtps2ph
  %res = call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %a0, i32 0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float>, i32) nounwind readonly


define <8 x i16> @test_x86_vcvtps2ph_256(<8 x float> %a0) {
  ; CHECK: vcvtps2ph
  %res = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %a0, i32 0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readonly
