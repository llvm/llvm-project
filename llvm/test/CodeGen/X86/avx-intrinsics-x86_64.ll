; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86-64 -mcpu=corei7 -mattr=avx | FileCheck %s

define i64 @test_x86_sse2_cvtsd2si64(<2 x double> %a0) {
  ; CHECK: vcvtsd2si
  %res = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone


define <2 x double> @test_x86_sse2_cvtsi642sd(<2 x double> %a0, i64 %a1) {
  ; CHECK: vcvtsi2sd
  %res = call <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double> %a0, i64 %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double>, i64) nounwind readnone


define i64 @test_x86_sse2_cvttsd2si64(<2 x double> %a0) {
  ; CHECK: vcvttsd2si
  %res = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone


define i64 @test_x86_sse_cvtss2si64(<4 x float> %a0) {
  ; CHECK: vcvtss2si
  %res = call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_cvtsi642ss(<4 x float> %a0, i64 %a1) {
  ; CHECK: vcvtsi2ss
  %res = call <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float> %a0, i64 %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float>, i64) nounwind readnone


define i64 @test_x86_sse_cvttss2si64(<4 x float> %a0) {
  ; CHECK: vcvttss2si
  %res = call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone


