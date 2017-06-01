; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-neon-syntax=apple -verify-machineinstrs -o - %s | FileCheck %s

; Test signed conversion.
; CHECK-LABEL: @test1
; CHECK: scvtf.2s v0, v0, #4
; CHECK: ret
define <2 x float> @test1(<2 x i32> %in) {
entry:
  %vcvt.i = sitofp <2 x i32> %in to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 16.0, float 16.0>
  ret <2 x float> %div.i
}

; Test unsigned conversion.
; CHECK-LABEL: @test2
; CHECK: ucvtf.2s v0, v0, #3
; CHECK: ret
define <2 x float> @test2(<2 x i32> %in) {
entry:
  %vcvt.i = uitofp <2 x i32> %in to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 8.0, float 8.0>
  ret <2 x float> %div.i
}

; Test which should not fold due to non-power of 2.
; CHECK-LABEL: @test3
; CHECK: scvtf.2s v0, v0
; CHECK: fmov.2s v1, #9.00000000
; CHECK: fdiv.2s v0, v0, v1
; CHECK: ret
define <2 x float> @test3(<2 x i32> %in) {
entry:
  %vcvt.i = sitofp <2 x i32> %in to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 9.0, float 9.0>
  ret <2 x float> %div.i
}

; Test which should not fold due to power of 2 out of range.
; CHECK-LABEL: @test4
; CHECK: scvtf.2s v0, v0
; CHECK: movi.2s v1, #80, lsl #24
; CHECK: fdiv.2s v0, v0, v1
; CHECK: ret
define <2 x float> @test4(<2 x i32> %in) {
entry:
  %vcvt.i = sitofp <2 x i32> %in to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 0x4200000000000000, float 0x4200000000000000>
  ret <2 x float> %div.i
}

; Test case where const is max power of 2 (i.e., 2^32).
; CHECK-LABEL: @test5
; CHECK: scvtf.2s v0, v0, #32
; CHECK: ret
define <2 x float> @test5(<2 x i32> %in) {
entry:
  %vcvt.i = sitofp <2 x i32> %in to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 0x41F0000000000000, float 0x41F0000000000000>
  ret <2 x float> %div.i
}

; Test quadword.
; CHECK-LABEL: @test6
; CHECK: scvtf.4s v0, v0, #2
; CHECK: ret
define <4 x float> @test6(<4 x i32> %in) {
entry:
  %vcvt.i = sitofp <4 x i32> %in to <4 x float>
  %div.i = fdiv <4 x float> %vcvt.i, <float 4.0, float 4.0, float 4.0, float 4.0>
  ret <4 x float> %div.i
}

; Test unsigned i16 to float
; CHECK-LABEL: @test7
; CHECK: ushll.4s  v0, v0, #0
; CHECK: ucvtf.4s  v0, v0, #1
; CHECK: ret
define <4 x float> @test7(<4 x i16> %in) {
  %conv = uitofp <4 x i16> %in to <4 x float>
  %shift = fdiv <4 x float> %conv, <float 2.0, float 2.0, float 2.0, float 2.0>
  ret <4 x float> %shift
}

; Test signed i16 to float
; CHECK-LABEL: @test8
; CHECK: sshll.4s v0, v0, #0
; CHECK: scvtf.4s v0, v0, #2
; CHECK: ret
define <4 x float> @test8(<4 x i16> %in) {
  %conv = sitofp <4 x i16> %in to <4 x float>
  %shift = fdiv <4 x float> %conv, <float 4.0, float 4.0, float 4.0, float 4.0>
  ret <4 x float> %shift
}

; Can't convert i64 to float.
; CHECK-LABEL: @test9
; CHECK: ucvtf.2d v0, v0
; CHECK: fcvtn v0.2s, v0.2d
; CHECK: movi.2s v1, #64, lsl #24
; CHECK: fdiv.2s v0, v0, v1
; CHECK: ret
define <2 x float> @test9(<2 x i64> %in) {
  %conv = uitofp <2 x i64> %in to <2 x float>
  %shift = fdiv <2 x float> %conv, <float 2.0, float 2.0>
  ret <2 x float> %shift
}

; CHECK-LABEL: @test10
; CHECK: ucvtf.2d v0, v0, #1
; CHECK: ret
define <2 x double> @test10(<2 x i64> %in) {
  %conv = uitofp <2 x i64> %in to <2 x double>
  %shift = fdiv <2 x double> %conv, <double 2.0, double 2.0>
  ret <2 x double> %shift
}
