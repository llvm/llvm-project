; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <2 x i32> @c1(<2 x float> %a) nounwind readnone ssp {
; CHECK: c1
; CHECK: fcvtzs.2s	v0, v0
; CHECK: ret
  %vcvt.i = fptosi <2 x float> %a to <2 x i32>
  ret <2 x i32> %vcvt.i
}

define <2 x i32> @c2(<2 x float> %a) nounwind readnone ssp {
; CHECK: c2
; CHECK: fcvtzu.2s	v0, v0
; CHECK: ret
  %vcvt.i = fptoui <2 x float> %a to <2 x i32>
  ret <2 x i32> %vcvt.i
}

define <4 x i32> @c3(<4 x float> %a) nounwind readnone ssp {
; CHECK: c3
; CHECK: fcvtzs.4s	v0, v0
; CHECK: ret
  %vcvt.i = fptosi <4 x float> %a to <4 x i32>
  ret <4 x i32> %vcvt.i
}

define <4 x i32> @c4(<4 x float> %a) nounwind readnone ssp {
; CHECK: c4
; CHECK: fcvtzu.4s	v0, v0
; CHECK: ret
  %vcvt.i = fptoui <4 x float> %a to <4 x i32>
  ret <4 x i32> %vcvt.i
}

