; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <8 x i8> @test_vget_high_s8(<16 x i8> %a) {
; CHECK-LABEL: test_vget_high_s8:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i8> %shuffle.i
}

define <4 x i16> @test_vget_high_s16(<8 x i16> %a) {
; CHECK-LABEL: test_vget_high_s16:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i16> %shuffle.i
}

define <2 x i32> @test_vget_high_s32(<4 x i32> %a) {
; CHECK-LABEL: test_vget_high_s32:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x i32> %shuffle.i
}

define <1 x i64> @test_vget_high_s64(<2 x i64> %a) {
; CHECK-LABEL: test_vget_high_s64:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> undef, <1 x i32> <i32 1>
  ret <1 x i64> %shuffle.i
}

define <8 x i8> @test_vget_high_u8(<16 x i8> %a) {
; CHECK-LABEL: test_vget_high_u8:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i8> %shuffle.i
}

define <4 x i16> @test_vget_high_u16(<8 x i16> %a) {
; CHECK-LABEL: test_vget_high_u16:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i16> %shuffle.i
}

define <2 x i32> @test_vget_high_u32(<4 x i32> %a) {
; CHECK-LABEL: test_vget_high_u32:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x i32> %shuffle.i
}

define <1 x i64> @test_vget_high_u64(<2 x i64> %a) {
; CHECK-LABEL: test_vget_high_u64:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> undef, <1 x i32> <i32 1>
  ret <1 x i64> %shuffle.i
}

define <1 x i64> @test_vget_high_p64(<2 x i64> %a) {
; CHECK-LABEL: test_vget_high_p64:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> undef, <1 x i32> <i32 1>
  ret <1 x i64> %shuffle.i
}

define <4 x i16> @test_vget_high_f16(<8 x i16> %a) {
; CHECK-LABEL: test_vget_high_f16:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i16> %shuffle.i
}

define <2 x float> @test_vget_high_f32(<4 x float> %a) {
; CHECK-LABEL: test_vget_high_f32:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x float> %shuffle.i
}

define <8 x i8> @test_vget_high_p8(<16 x i8> %a) {
; CHECK-LABEL: test_vget_high_p8:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i8> %shuffle.i
}

define <4 x i16> @test_vget_high_p16(<8 x i16> %a) {
; CHECK-LABEL: test_vget_high_p16:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i16> %shuffle.i
}

define <1 x double> @test_vget_high_f64(<2 x double> %a) {
; CHECK-LABEL: test_vget_high_f64:
; CHECK: dup d0, {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x double> %a, <2 x double> undef, <1 x i32> <i32 1>
  ret <1 x double> %shuffle.i
}

define <8 x i8> @test_vget_low_s8(<16 x i8> %a) {
; CHECK-LABEL: test_vget_low_s8:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %shuffle.i
}

define <4 x i16> @test_vget_low_s16(<8 x i16> %a) {
; CHECK-LABEL: test_vget_low_s16:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %shuffle.i
}

define <2 x i32> @test_vget_low_s32(<4 x i32> %a) {
; CHECK-LABEL: test_vget_low_s32:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x i32> %shuffle.i
}

define <1 x i64> @test_vget_low_s64(<2 x i64> %a) {
; CHECK-LABEL: test_vget_low_s64:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> undef, <1 x i32> zeroinitializer
  ret <1 x i64> %shuffle.i
}

define <8 x i8> @test_vget_low_u8(<16 x i8> %a) {
; CHECK-LABEL: test_vget_low_u8:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %shuffle.i
}

define <4 x i16> @test_vget_low_u16(<8 x i16> %a) {
; CHECK-LABEL: test_vget_low_u16:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %shuffle.i
}

define <2 x i32> @test_vget_low_u32(<4 x i32> %a) {
; CHECK-LABEL: test_vget_low_u32:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x i32> %shuffle.i
}

define <1 x i64> @test_vget_low_u64(<2 x i64> %a) {
; CHECK-LABEL: test_vget_low_u64:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> undef, <1 x i32> zeroinitializer
  ret <1 x i64> %shuffle.i
}

define <1 x i64> @test_vget_low_p64(<2 x i64> %a) {
; CHECK-LABEL: test_vget_low_p64:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> undef, <1 x i32> zeroinitializer
  ret <1 x i64> %shuffle.i
}

define <4 x i16> @test_vget_low_f16(<8 x i16> %a) {
; CHECK-LABEL: test_vget_low_f16:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %shuffle.i
}

define <2 x float> @test_vget_low_f32(<4 x float> %a) {
; CHECK-LABEL: test_vget_low_f32:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %shuffle.i
}

define <8 x i8> @test_vget_low_p8(<16 x i8> %a) {
; CHECK-LABEL: test_vget_low_p8:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %shuffle.i
}

define <4 x i16> @test_vget_low_p16(<8 x i16> %a) {
; CHECK-LABEL: test_vget_low_p16:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %shuffle.i
}

define <1 x double> @test_vget_low_f64(<2 x double> %a) {
; CHECK-LABEL: test_vget_low_f64:
; CHECK: ret
entry:
  %shuffle.i = shufflevector <2 x double> %a, <2 x double> undef, <1 x i32> zeroinitializer
  ret <1 x double> %shuffle.i
}
