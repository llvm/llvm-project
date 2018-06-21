; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-neon-syntax=apple -verify-machineinstrs -o - %s | FileCheck %s

; CHECK-LABEL: test1
; CHECK-NOT: fmul.2s
; CHECK: fcvtzs.2s v0, v0, #4
; CHECK: ret
define <2 x i32> @test1(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 16.000000e+00, float 16.000000e+00>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; CHECK-LABEL: test2
; CHECK-NOT: fmul.4s
; CHECK: fcvtzs.4s v0, v0, #3
; CHECK: ret
define <4 x i32> @test2(<4 x float> %f) {
  %mul.i = fmul <4 x float> %f, <float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00>
  %vcvt.i = fptosi <4 x float> %mul.i to <4 x i32>
  ret <4 x i32> %vcvt.i
}

; CHECK-LABEL: test3
; CHECK-NOT: fmul.2d
; CHECK: fcvtzs.2d v0, v0, #5
; CHECK: ret
define <2 x i64> @test3(<2 x double> %d) {
  %mul.i = fmul <2 x double> %d, <double 32.000000e+00, double 32.000000e+00>
  %vcvt.i = fptosi <2 x double> %mul.i to <2 x i64>
  ret <2 x i64> %vcvt.i
}

; Truncate double to i32
; CHECK-LABEL: test4
; CHECK-NOT: fmul.2d v0, v0, #4
; CHECK: fcvtzs.2d v0, v0
; CHECK: xtn.2s
; CHECK: ret
define <2 x i32> @test4(<2 x double> %d) {
  %mul.i = fmul <2 x double> %d, <double 16.000000e+00, double 16.000000e+00>
  %vcvt.i = fptosi <2 x double> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Truncate float to i16
; CHECK-LABEL: test5
; CHECK-NOT: fmul.2s
; CHECK: fcvtzs.2s v0, v0, #4
; CHECK: ret
define <2 x i16> @test5(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 16.000000e+00, float 16.000000e+00>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i16>
  ret <2 x i16> %vcvt.i
}

; Don't convert float to i64
; CHECK-LABEL: test6
; CHECK: fmov.2s v1, #16.00000000
; CHECK: fmul.2s v0, v0, v1
; CHECK: fcvtl v0.2d, v0.2s
; CHECK: fcvtzs.2d v0, v0
; CHECK: ret
define <2 x i64> @test6(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 16.000000e+00, float 16.000000e+00>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i64>
  ret <2 x i64> %vcvt.i
}

; Check unsigned conversion.
; CHECK-LABEL: test7
; CHECK-NOT: fmul.2s
; CHECK: fcvtzu.2s v0, v0, #4
; CHECK: ret
define <2 x i32> @test7(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 16.000000e+00, float 16.000000e+00>
  %vcvt.i = fptoui <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test which should not fold due to non-power of 2.
; CHECK-LABEL: test8
; CHECK: fmov.2s v1, #17.00000000
; CHECK: fmul.2s v0, v0, v1
; CHECK: fcvtzu.2s v0, v0
; CHECK: ret
define <2 x i32> @test8(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 17.000000e+00, float 17.000000e+00>
  %vcvt.i = fptoui <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test which should not fold due to non-matching power of 2.
; CHECK-LABEL: test9
; CHECK: fmul.2s v0, v0, v1
; CHECK: fcvtzu.2s v0, v0
; CHECK: ret
define <2 x i32> @test9(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 16.000000e+00, float 8.000000e+00>
  %vcvt.i = fptoui <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Don't combine all undefs.
; CHECK-LABEL: test10
; CHECK: fmul.2s v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CHECK: fcvtzu.2s v{{[0-9]+}}, v{{[0-9]+}}
; CHECK: ret
define <2 x i32> @test10(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float undef, float undef>
  %vcvt.i = fptoui <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Combine if mix of undef and pow2.
; CHECK-LABEL: test11
; CHECK: fcvtzu.2s v0, v0, #3
; CHECK: ret
define <2 x i32> @test11(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float undef, float 8.000000e+00>
  %vcvt.i = fptoui <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Don't combine when multiplied by 0.0.
; CHECK-LABEL: test12
; CHECK: fmul.2s v0, v0, v1
; CHECK: fcvtzs.2s v0, v0
; CHECK: ret
define <2 x i32> @test12(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 0.000000e+00, float 0.000000e+00>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test which should not fold due to power of 2 out of range (i.e., 2^33).
; CHECK-LABEL: test13
; CHECK: fmul.2s v0, v0, v1
; CHECK: fcvtzs.2s v0, v0
; CHECK: ret
define <2 x i32> @test13(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 0x4200000000000000, float 0x4200000000000000>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test case where const is max power of 2 (i.e., 2^32).
; CHECK-LABEL: test14
; CHECK: fcvtzs.2s v0, v0, #32
; CHECK: ret
define <2 x i32> @test14(<2 x float> %f) {
  %mul.i = fmul <2 x float> %f, <float 0x41F0000000000000, float 0x41F0000000000000>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; CHECK-LABEL: test_illegal_fp_to_int:
; CHECK: fcvtzs.4s v0, v0, #2
define <3 x i32> @test_illegal_fp_to_int(<3 x float> %in) {
  %scale = fmul <3 x float> %in, <float 4.0, float 4.0, float 4.0>
  %val = fptosi <3 x float> %scale to <3 x i32>
  ret <3 x i32> %val
}
