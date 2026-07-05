; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s

; Test signed conversion.
; CHECK-LABEL: @t0
; CHECK: vcvt.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}, #2
; CHECK: bx lr
define <2 x i32> @t0(<2 x float> %in) {
  %mul.i = fmul <2 x float> %in, <float 4.0, float 4.0>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test unsigned conversion.
; CHECK-LABEL: @t1
; CHECK: vcvt.u32.f32 d{{[0-9]+}}, d{{[0-9]+}}, #3
; CHECK: bx lr
define <2 x i32> @t1(<2 x float> %in) {
  %mul.i = fmul <2 x float> %in, <float 8.0, float 8.0>
  %vcvt.i = fptoui <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test which should not fold due to non-power of 2.
; CHECK-LABEL: @t2
; CHECK: vmul
; CHECK: vcvt.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: bx lr
define <2 x i32> @t2(<2 x float> %in) {
entry:
  %mul.i = fmul <2 x float> %in, <float 0x401B333340000000, float 0x401B333340000000>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test which should not fold due to power of 2 out of range.
; CHECK-LABEL: @t3
; CHECK: vmul
; CHECK: vcvt.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: bx lr
define <2 x i32> @t3(<2 x float> %in) {
  %mul.i = fmul <2 x float> %in, <float 0x4200000000000000, float 0x4200000000000000>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test which case where const is max power of 2 (i.e., 2^32).
; CHECK-LABEL: @t4
; CHECK: vcvt.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}, #32
; CHECK: bx lr
define <2 x i32> @t4(<2 x float> %in) {
  %mul.i = fmul <2 x float> %in, <float 0x41F0000000000000, float 0x41F0000000000000>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  ret <2 x i32> %vcvt.i
}

; Test quadword.
; CHECK-LABEL: @t5
; CHECK: vcvt.s32.f32 q{{[0-9]+}}, q{{[0-9]+}}, #3
; CHECK: bx lr
define <4 x i32> @t5(<4 x float> %in) {
  %mul.i = fmul <4 x float> %in, <float 8.0, float 8.0, float 8.0, float 8.0>
  %vcvt.i = fptosi <4 x float> %mul.i to <4 x i32>
  ret <4 x i32> %vcvt.i
}

; CHECK-LABEL: test_illegal_fp_to_int:
; CHECK: vcvt.s32.f32 {{q[0-9]+}}, {{q[0-9]+}}, #2
define <3 x i32> @test_illegal_fp_to_int(<3 x float> %in) {
  %scale = fmul <3 x float> %in, <float 4.0, float 4.0, float 4.0>
  %val = fptosi <3 x float> %scale to <3 x i32>
  ret <3 x i32> %val
}
