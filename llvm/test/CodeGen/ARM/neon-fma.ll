; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -mcpu=swift | FileCheck %s

; CHECK: test_v2f32
; CHECK: vfma.f32 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}

define <2 x float> @test_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind readnone ssp {
entry:
  %call = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind readnone
  ret <2 x float> %call
}

; CHECK: test_v4f32
; CHECK: vfma.f32 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}

define <4 x float> @test_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c) nounwind readnone ssp {
entry:
  %call = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c) nounwind readnone
  ret <4 x float> %call
}

declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone
