; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=corei7-avx -mattr=+avx | FileCheck %s

;CHECK: foo1_8
;CHECK: vcvtdq2ps
;CHECK: ret
define <8 x float> @foo1_8(<8 x i8> %src) {
  %res = sitofp <8 x i8> %src to <8 x float>
  ret <8 x float> %res
}

;CHECK: foo1_4
;CHECK: vcvtdq2ps
;CHECK: ret
define <4 x float> @foo1_4(<4 x i8> %src) {
  %res = sitofp <4 x i8> %src to <4 x float>
  ret <4 x float> %res
}

;CHECK: foo2_8
;CHECK: vcvtdq2ps
;CHECK: ret
define <8 x float> @foo2_8(<8 x i8> %src) {
  %res = uitofp <8 x i8> %src to <8 x float>
  ret <8 x float> %res
}

;CHECK: foo2_4
;CHECK: vcvtdq2ps
;CHECK: ret
define <4 x float> @foo2_4(<4 x i8> %src) {
  %res = uitofp <4 x i8> %src to <4 x float>
  ret <4 x float> %res
}

;CHECK: foo3_8
;CHECK: vcvttps2dq
;CHECK: ret
define <8 x i8> @foo3_8(<8 x float> %src) {
  %res = fptosi <8 x float> %src to <8 x i8>
  ret <8 x i8> %res
}
;CHECK: foo3_4
;CHECK: vcvttps2dq
;CHECK: ret
define <4 x i8> @foo3_4(<4 x float> %src) {
  %res = fptosi <4 x float> %src to <4 x i8>
  ret <4 x i8> %res
}

