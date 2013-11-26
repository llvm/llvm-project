; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

%struct.int8x8x2_t = type { [2 x <8 x i8>] }
%struct.int16x4x2_t = type { [2 x <4 x i16>] }
%struct.int32x2x2_t = type { [2 x <2 x i32>] }
%struct.uint8x8x2_t = type { [2 x <8 x i8>] }
%struct.uint16x4x2_t = type { [2 x <4 x i16>] }
%struct.uint32x2x2_t = type { [2 x <2 x i32>] }
%struct.float32x2x2_t = type { [2 x <2 x float>] }
%struct.poly8x8x2_t = type { [2 x <8 x i8>] }
%struct.poly16x4x2_t = type { [2 x <4 x i16>] }
%struct.int8x16x2_t = type { [2 x <16 x i8>] }
%struct.int16x8x2_t = type { [2 x <8 x i16>] }
%struct.int32x4x2_t = type { [2 x <4 x i32>] }
%struct.uint8x16x2_t = type { [2 x <16 x i8>] }
%struct.uint16x8x2_t = type { [2 x <8 x i16>] }
%struct.uint32x4x2_t = type { [2 x <4 x i32>] }
%struct.float32x4x2_t = type { [2 x <4 x float>] }
%struct.poly8x16x2_t = type { [2 x <16 x i8>] }
%struct.poly16x8x2_t = type { [2 x <8 x i16>] }

define <8 x i8> @test_vuzp1_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp1_s8:
; CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vuzp1q_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzp1q_s8:
; CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vuzp1_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp1_s16:
; CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vuzp1q_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzp1q_s16:
; CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vuzp1_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vuzp1_s32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vuzp1q_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vuzp1q_s32:
; CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vuzp1q_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vuzp1q_s64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle.i
}

define <8 x i8> @test_vuzp1_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp1_u8:
; CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vuzp1q_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzp1q_u8:
; CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vuzp1_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp1_u16:
; CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vuzp1q_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzp1q_u16:
; CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vuzp1_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vuzp1_u32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vuzp1q_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vuzp1q_u32:
; CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vuzp1q_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vuzp1q_u64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle.i
}

define <2 x float> @test_vuzp1_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vuzp1_f32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x float> %shuffle.i
}

define <4 x float> @test_vuzp1q_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vuzp1q_f32:
; CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x float> %shuffle.i
}

define <2 x double> @test_vuzp1q_f64(<2 x double> %a, <2 x double> %b) {
; CHECK: test_vuzp1q_f64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %shuffle.i
}

define <8 x i8> @test_vuzp1_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp1_p8:
; CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vuzp1q_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzp1q_p8:
; CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vuzp1_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp1_p16:
; CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vuzp1q_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzp1q_p16:
; CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ret <8 x i16> %shuffle.i
}

define <8 x i8> @test_vuzp2_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp2_s8:
; CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vuzp2q_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzp2q_s8:
; CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vuzp2_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp2_s16:
; CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vuzp2q_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzp2q_s16:
; CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vuzp2_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vuzp2_s32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vuzp2q_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vuzp2q_s32:
; CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vuzp2q_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vuzp2q_s64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle.i
}

define <8 x i8> @test_vuzp2_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp2_u8:
; CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vuzp2q_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzp2q_u8:
; CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vuzp2_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp2_u16:
; CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vuzp2q_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzp2q_u16:
; CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vuzp2_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vuzp2_u32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vuzp2q_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vuzp2q_u32:
; CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vuzp2q_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vuzp2q_u64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle.i
}

define <2 x float> @test_vuzp2_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vuzp2_f32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x float> %shuffle.i
}

define <4 x float> @test_vuzp2q_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vuzp2q_f32:
; CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x float> %shuffle.i
}

define <2 x double> @test_vuzp2q_f64(<2 x double> %a, <2 x double> %b) {
; CHECK: test_vuzp2q_f64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %shuffle.i
}

define <8 x i8> @test_vuzp2_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp2_p8:
; CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vuzp2q_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzp2q_p8:
; CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vuzp2_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp2_p16:
; CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vuzp2q_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzp2q_p16:
; CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret <8 x i16> %shuffle.i
}

define <8 x i8> @test_vzip1_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip1_s8:
; CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vzip1q_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzip1q_s8:
; CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vzip1_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip1_s16:
; CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vzip1q_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzip1q_s16:
; CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vzip1_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vzip1_s32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vzip1q_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vzip1q_s32:
; CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vzip1q_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vzip1q_s64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle.i
}

define <8 x i8> @test_vzip1_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip1_u8:
; CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vzip1q_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzip1q_u8:
; CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vzip1_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip1_u16:
; CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vzip1q_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzip1q_u16:
; CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vzip1_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vzip1_u32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vzip1q_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vzip1q_u32:
; CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vzip1q_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vzip1q_u64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle.i
}

define <2 x float> @test_vzip1_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vzip1_f32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x float> %shuffle.i
}

define <4 x float> @test_vzip1q_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vzip1q_f32:
; CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x float> %shuffle.i
}

define <2 x double> @test_vzip1q_f64(<2 x double> %a, <2 x double> %b) {
; CHECK: test_vzip1q_f64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %shuffle.i
}

define <8 x i8> @test_vzip1_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip1_p8:
; CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vzip1q_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzip1q_p8:
; CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vzip1_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip1_p16:
; CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vzip1q_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzip1q_p16:
; CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i16> %shuffle.i
}

define <8 x i8> @test_vzip2_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip2_s8:
; CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vzip2q_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzip2q_s8:
; CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vzip2_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip2_s16:
; CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vzip2q_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzip2q_s16:
; CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vzip2_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vzip2_s32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vzip2q_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vzip2q_s32:
; CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vzip2q_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vzip2q_s64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle.i
}

define <8 x i8> @test_vzip2_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip2_u8:
; CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vzip2q_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzip2q_u8:
; CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vzip2_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip2_u16:
; CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vzip2q_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzip2q_u16:
; CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vzip2_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vzip2_u32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vzip2q_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vzip2q_u32:
; CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vzip2q_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vzip2q_u64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle.i
}

define <2 x float> @test_vzip2_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vzip2_f32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x float> %shuffle.i
}

define <4 x float> @test_vzip2q_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vzip2q_f32:
; CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x float> %shuffle.i
}

define <2 x double> @test_vzip2q_f64(<2 x double> %a, <2 x double> %b) {
; CHECK: test_vzip2q_f64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %shuffle.i
}

define <8 x i8> @test_vzip2_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip2_p8:
; CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vzip2q_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzip2q_p8:
; CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vzip2_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip2_p16:
; CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vzip2q_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzip2q_p16:
; CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i16> %shuffle.i
}

define <8 x i8> @test_vtrn1_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn1_s8:
; CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vtrn1q_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrn1q_s8:
; CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vtrn1_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn1_s16:
; CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vtrn1q_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrn1q_s16:
; CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vtrn1_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vtrn1_s32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vtrn1q_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vtrn1q_s32:
; CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vtrn1q_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vtrn1q_s64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle.i
}

define <8 x i8> @test_vtrn1_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn1_u8:
; CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vtrn1q_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrn1q_u8:
; CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vtrn1_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn1_u16:
; CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vtrn1q_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrn1q_u16:
; CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vtrn1_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vtrn1_u32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vtrn1q_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vtrn1q_u32:
; CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vtrn1q_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vtrn1q_u64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle.i
}

define <2 x float> @test_vtrn1_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vtrn1_f32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
entry:
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x float> %shuffle.i
}

define <4 x float> @test_vtrn1q_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vtrn1q_f32:
; CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x float> %shuffle.i
}

define <2 x double> @test_vtrn1q_f64(<2 x double> %a, <2 x double> %b) {
; CHECK: test_vtrn1q_f64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %shuffle.i = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %shuffle.i
}

define <8 x i8> @test_vtrn1_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn1_p8:
; CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vtrn1q_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrn1q_p8:
; CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vtrn1_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn1_p16:
; CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vtrn1q_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrn1q_p16:
; CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i16> %shuffle.i
}

define <8 x i8> @test_vtrn2_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn2_s8:
; CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vtrn2q_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrn2q_s8:
; CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vtrn2_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn2_s16:
; CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vtrn2q_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrn2q_s16:
; CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vtrn2_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vtrn2_s32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vtrn2q_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vtrn2q_s32:
; CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vtrn2q_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vtrn2q_s64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle.i
}

define <8 x i8> @test_vtrn2_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn2_u8:
; CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vtrn2q_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrn2q_u8:
; CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vtrn2_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn2_u16:
; CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vtrn2q_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrn2q_u16:
; CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i16> %shuffle.i
}

define <2 x i32> @test_vtrn2_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vtrn2_u32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %shuffle.i
}

define <4 x i32> @test_vtrn2q_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vtrn2q_u32:
; CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i32> %shuffle.i
}

define <2 x i64> @test_vtrn2q_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vtrn2q_u64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle.i
}

define <2 x float> @test_vtrn2_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vtrn2_f32:
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x float> %shuffle.i
}

define <4 x float> @test_vtrn2q_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vtrn2q_f32:
; CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x float> %shuffle.i
}

define <2 x double> @test_vtrn2q_f64(<2 x double> %a, <2 x double> %b) {
; CHECK: test_vtrn2q_f64:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
entry:
  %shuffle.i = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %shuffle.i
}

define <8 x i8> @test_vtrn2_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn2_p8:
; CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vtrn2q_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrn2q_p8:
; CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  ret <16 x i8> %shuffle.i
}

define <4 x i16> @test_vtrn2_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn2_p16:
; CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i16> %shuffle.i
}

define <8 x i16> @test_vtrn2q_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrn2q_p16:
; CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i16> %shuffle.i
}

define %struct.int8x8x2_t @test_vuzp_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp_s8:
; CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vuzp.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vuzp1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %.fca.0.0.insert = insertvalue %struct.int8x8x2_t undef, <8 x i8> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x2_t %.fca.0.0.insert, <8 x i8> %vuzp1.i, 0, 1
  ret %struct.int8x8x2_t %.fca.0.1.insert
}

define %struct.int16x4x2_t @test_vuzp_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp_s16:
; CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vuzp.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %vuzp1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %.fca.0.0.insert = insertvalue %struct.int16x4x2_t undef, <4 x i16> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x2_t %.fca.0.0.insert, <4 x i16> %vuzp1.i, 0, 1
  ret %struct.int16x4x2_t %.fca.0.1.insert
}

define %struct.int32x2x2_t @test_vuzp_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vuzp_s32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vuzp.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  %vuzp1.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.int32x2x2_t undef, <2 x i32> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x2_t %.fca.0.0.insert, <2 x i32> %vuzp1.i, 0, 1
  ret %struct.int32x2x2_t %.fca.0.1.insert
}

define %struct.uint8x8x2_t @test_vuzp_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp_u8:
; CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vuzp.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vuzp1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %.fca.0.0.insert = insertvalue %struct.uint8x8x2_t undef, <8 x i8> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint8x8x2_t %.fca.0.0.insert, <8 x i8> %vuzp1.i, 0, 1
  ret %struct.uint8x8x2_t %.fca.0.1.insert
}

define %struct.uint16x4x2_t @test_vuzp_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp_u16:
; CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vuzp.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %vuzp1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %.fca.0.0.insert = insertvalue %struct.uint16x4x2_t undef, <4 x i16> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint16x4x2_t %.fca.0.0.insert, <4 x i16> %vuzp1.i, 0, 1
  ret %struct.uint16x4x2_t %.fca.0.1.insert
}

define %struct.uint32x2x2_t @test_vuzp_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vuzp_u32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vuzp.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  %vuzp1.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.uint32x2x2_t undef, <2 x i32> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint32x2x2_t %.fca.0.0.insert, <2 x i32> %vuzp1.i, 0, 1
  ret %struct.uint32x2x2_t %.fca.0.1.insert
}

define %struct.float32x2x2_t @test_vuzp_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vuzp_f32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vuzp.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
  %vuzp1.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.float32x2x2_t undef, <2 x float> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x2_t %.fca.0.0.insert, <2 x float> %vuzp1.i, 0, 1
  ret %struct.float32x2x2_t %.fca.0.1.insert
}

define %struct.poly8x8x2_t @test_vuzp_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vuzp_p8:
; CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vuzp.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vuzp1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %.fca.0.0.insert = insertvalue %struct.poly8x8x2_t undef, <8 x i8> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly8x8x2_t %.fca.0.0.insert, <8 x i8> %vuzp1.i, 0, 1
  ret %struct.poly8x8x2_t %.fca.0.1.insert
}

define %struct.poly16x4x2_t @test_vuzp_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vuzp_p16:
; CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vuzp.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %vuzp1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %.fca.0.0.insert = insertvalue %struct.poly16x4x2_t undef, <4 x i16> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly16x4x2_t %.fca.0.0.insert, <4 x i16> %vuzp1.i, 0, 1
  ret %struct.poly16x4x2_t %.fca.0.1.insert
}

define %struct.int8x16x2_t @test_vuzpq_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzpq_s8:
; CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vuzp.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %vuzp1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %.fca.0.0.insert = insertvalue %struct.int8x16x2_t undef, <16 x i8> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x2_t %.fca.0.0.insert, <16 x i8> %vuzp1.i, 0, 1
  ret %struct.int8x16x2_t %.fca.0.1.insert
}

define %struct.int16x8x2_t @test_vuzpq_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzpq_s16:
; CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vuzp.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vuzp1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %.fca.0.0.insert = insertvalue %struct.int16x8x2_t undef, <8 x i16> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x2_t %.fca.0.0.insert, <8 x i16> %vuzp1.i, 0, 1
  ret %struct.int16x8x2_t %.fca.0.1.insert
}

define %struct.int32x4x2_t @test_vuzpq_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vuzpq_s32:
; CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vuzp.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %vuzp1.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %.fca.0.0.insert = insertvalue %struct.int32x4x2_t undef, <4 x i32> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x2_t %.fca.0.0.insert, <4 x i32> %vuzp1.i, 0, 1
  ret %struct.int32x4x2_t %.fca.0.1.insert
}

define %struct.uint8x16x2_t @test_vuzpq_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzpq_u8:
; CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vuzp.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %vuzp1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %.fca.0.0.insert = insertvalue %struct.uint8x16x2_t undef, <16 x i8> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint8x16x2_t %.fca.0.0.insert, <16 x i8> %vuzp1.i, 0, 1
  ret %struct.uint8x16x2_t %.fca.0.1.insert
}

define %struct.uint16x8x2_t @test_vuzpq_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzpq_u16:
; CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vuzp.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vuzp1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %.fca.0.0.insert = insertvalue %struct.uint16x8x2_t undef, <8 x i16> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint16x8x2_t %.fca.0.0.insert, <8 x i16> %vuzp1.i, 0, 1
  ret %struct.uint16x8x2_t %.fca.0.1.insert
}

define %struct.uint32x4x2_t @test_vuzpq_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vuzpq_u32:
; CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vuzp.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %vuzp1.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %.fca.0.0.insert = insertvalue %struct.uint32x4x2_t undef, <4 x i32> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint32x4x2_t %.fca.0.0.insert, <4 x i32> %vuzp1.i, 0, 1
  ret %struct.uint32x4x2_t %.fca.0.1.insert
}

define %struct.float32x4x2_t @test_vuzpq_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vuzpq_f32:
; CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vuzp.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %vuzp1.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %.fca.0.0.insert = insertvalue %struct.float32x4x2_t undef, <4 x float> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x2_t %.fca.0.0.insert, <4 x float> %vuzp1.i, 0, 1
  ret %struct.float32x4x2_t %.fca.0.1.insert
}

define %struct.poly8x16x2_t @test_vuzpq_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vuzpq_p8:
; CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vuzp.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %vuzp1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %.fca.0.0.insert = insertvalue %struct.poly8x16x2_t undef, <16 x i8> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly8x16x2_t %.fca.0.0.insert, <16 x i8> %vuzp1.i, 0, 1
  ret %struct.poly8x16x2_t %.fca.0.1.insert
}

define %struct.poly16x8x2_t @test_vuzpq_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vuzpq_p16:
; CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vuzp.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vuzp1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %.fca.0.0.insert = insertvalue %struct.poly16x8x2_t undef, <8 x i16> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly16x8x2_t %.fca.0.0.insert, <8 x i16> %vuzp1.i, 0, 1
  ret %struct.poly16x8x2_t %.fca.0.1.insert
}

define %struct.int8x8x2_t @test_vzip_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip_s8:
; CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vzip.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %vzip1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.int8x8x2_t undef, <8 x i8> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x2_t %.fca.0.0.insert, <8 x i8> %vzip1.i, 0, 1
  ret %struct.int8x8x2_t %.fca.0.1.insert
}

define %struct.int16x4x2_t @test_vzip_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip_s16:
; CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vzip.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %vzip1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.int16x4x2_t undef, <4 x i16> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x2_t %.fca.0.0.insert, <4 x i16> %vzip1.i, 0, 1
  ret %struct.int16x4x2_t %.fca.0.1.insert
}

define %struct.int32x2x2_t @test_vzip_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vzip_s32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vzip.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  %vzip1.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.int32x2x2_t undef, <2 x i32> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x2_t %.fca.0.0.insert, <2 x i32> %vzip1.i, 0, 1
  ret %struct.int32x2x2_t %.fca.0.1.insert
}

define %struct.uint8x8x2_t @test_vzip_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip_u8:
; CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vzip.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %vzip1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.uint8x8x2_t undef, <8 x i8> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint8x8x2_t %.fca.0.0.insert, <8 x i8> %vzip1.i, 0, 1
  ret %struct.uint8x8x2_t %.fca.0.1.insert
}

define %struct.uint16x4x2_t @test_vzip_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip_u16:
; CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vzip.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %vzip1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.uint16x4x2_t undef, <4 x i16> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint16x4x2_t %.fca.0.0.insert, <4 x i16> %vzip1.i, 0, 1
  ret %struct.uint16x4x2_t %.fca.0.1.insert
}

define %struct.uint32x2x2_t @test_vzip_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vzip_u32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vzip.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  %vzip1.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.uint32x2x2_t undef, <2 x i32> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint32x2x2_t %.fca.0.0.insert, <2 x i32> %vzip1.i, 0, 1
  ret %struct.uint32x2x2_t %.fca.0.1.insert
}

define %struct.float32x2x2_t @test_vzip_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vzip_f32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vzip.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
  %vzip1.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.float32x2x2_t undef, <2 x float> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x2_t %.fca.0.0.insert, <2 x float> %vzip1.i, 0, 1
  ret %struct.float32x2x2_t %.fca.0.1.insert
}

define %struct.poly8x8x2_t @test_vzip_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vzip_p8:
; CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vzip.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %vzip1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.poly8x8x2_t undef, <8 x i8> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly8x8x2_t %.fca.0.0.insert, <8 x i8> %vzip1.i, 0, 1
  ret %struct.poly8x8x2_t %.fca.0.1.insert
}

define %struct.poly16x4x2_t @test_vzip_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vzip_p16:
; CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vzip.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %vzip1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.poly16x4x2_t undef, <4 x i16> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly16x4x2_t %.fca.0.0.insert, <4 x i16> %vzip1.i, 0, 1
  ret %struct.poly16x4x2_t %.fca.0.1.insert
}

define %struct.int8x16x2_t @test_vzipq_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzipq_s8:
; CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vzip.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  %vzip1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %.fca.0.0.insert = insertvalue %struct.int8x16x2_t undef, <16 x i8> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x2_t %.fca.0.0.insert, <16 x i8> %vzip1.i, 0, 1
  ret %struct.int8x16x2_t %.fca.0.1.insert
}

define %struct.int16x8x2_t @test_vzipq_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzipq_s16:
; CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vzip.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %vzip1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.int16x8x2_t undef, <8 x i16> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x2_t %.fca.0.0.insert, <8 x i16> %vzip1.i, 0, 1
  ret %struct.int16x8x2_t %.fca.0.1.insert
}

define %struct.int32x4x2_t @test_vzipq_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vzipq_s32:
; CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vzip.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %vzip1.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.int32x4x2_t undef, <4 x i32> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x2_t %.fca.0.0.insert, <4 x i32> %vzip1.i, 0, 1
  ret %struct.int32x4x2_t %.fca.0.1.insert
}

define %struct.uint8x16x2_t @test_vzipq_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzipq_u8:
; CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vzip.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  %vzip1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %.fca.0.0.insert = insertvalue %struct.uint8x16x2_t undef, <16 x i8> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint8x16x2_t %.fca.0.0.insert, <16 x i8> %vzip1.i, 0, 1
  ret %struct.uint8x16x2_t %.fca.0.1.insert
}

define %struct.uint16x8x2_t @test_vzipq_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzipq_u16:
; CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vzip.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %vzip1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.uint16x8x2_t undef, <8 x i16> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint16x8x2_t %.fca.0.0.insert, <8 x i16> %vzip1.i, 0, 1
  ret %struct.uint16x8x2_t %.fca.0.1.insert
}

define %struct.uint32x4x2_t @test_vzipq_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vzipq_u32:
; CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vzip.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %vzip1.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.uint32x4x2_t undef, <4 x i32> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint32x4x2_t %.fca.0.0.insert, <4 x i32> %vzip1.i, 0, 1
  ret %struct.uint32x4x2_t %.fca.0.1.insert
}

define %struct.float32x4x2_t @test_vzipq_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vzipq_f32:
; CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vzip.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %vzip1.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.float32x4x2_t undef, <4 x float> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x2_t %.fca.0.0.insert, <4 x float> %vzip1.i, 0, 1
  ret %struct.float32x4x2_t %.fca.0.1.insert
}

define %struct.poly8x16x2_t @test_vzipq_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vzipq_p8:
; CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vzip.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  %vzip1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  %.fca.0.0.insert = insertvalue %struct.poly8x16x2_t undef, <16 x i8> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly8x16x2_t %.fca.0.0.insert, <16 x i8> %vzip1.i, 0, 1
  ret %struct.poly8x16x2_t %.fca.0.1.insert
}

define %struct.poly16x8x2_t @test_vzipq_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vzipq_p16:
; CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vzip.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %vzip1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.poly16x8x2_t undef, <8 x i16> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly16x8x2_t %.fca.0.0.insert, <8 x i16> %vzip1.i, 0, 1
  ret %struct.poly16x8x2_t %.fca.0.1.insert
}

define %struct.int8x8x2_t @test_vtrn_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn_s8:
; CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vtrn.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %vtrn1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.int8x8x2_t undef, <8 x i8> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x2_t %.fca.0.0.insert, <8 x i8> %vtrn1.i, 0, 1
  ret %struct.int8x8x2_t %.fca.0.1.insert
}

define %struct.int16x4x2_t @test_vtrn_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn_s16:
; CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vtrn.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %vtrn1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.int16x4x2_t undef, <4 x i16> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x2_t %.fca.0.0.insert, <4 x i16> %vtrn1.i, 0, 1
  ret %struct.int16x4x2_t %.fca.0.1.insert
}

define %struct.int32x2x2_t @test_vtrn_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vtrn_s32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vtrn.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  %vtrn1.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.int32x2x2_t undef, <2 x i32> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x2_t %.fca.0.0.insert, <2 x i32> %vtrn1.i, 0, 1
  ret %struct.int32x2x2_t %.fca.0.1.insert
}

define %struct.uint8x8x2_t @test_vtrn_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn_u8:
; CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vtrn.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %vtrn1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.uint8x8x2_t undef, <8 x i8> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint8x8x2_t %.fca.0.0.insert, <8 x i8> %vtrn1.i, 0, 1
  ret %struct.uint8x8x2_t %.fca.0.1.insert
}

define %struct.uint16x4x2_t @test_vtrn_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn_u16:
; CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vtrn.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %vtrn1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.uint16x4x2_t undef, <4 x i16> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint16x4x2_t %.fca.0.0.insert, <4 x i16> %vtrn1.i, 0, 1
  ret %struct.uint16x4x2_t %.fca.0.1.insert
}

define %struct.uint32x2x2_t @test_vtrn_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vtrn_u32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vtrn.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
  %vtrn1.i = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.uint32x2x2_t undef, <2 x i32> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint32x2x2_t %.fca.0.0.insert, <2 x i32> %vtrn1.i, 0, 1
  ret %struct.uint32x2x2_t %.fca.0.1.insert
}

define %struct.float32x2x2_t @test_vtrn_f32(<2 x float> %a, <2 x float> %b) {
; CHECK: test_vtrn_f32:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
entry:
  %vtrn.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
  %vtrn1.i = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
  %.fca.0.0.insert = insertvalue %struct.float32x2x2_t undef, <2 x float> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x2_t %.fca.0.0.insert, <2 x float> %vtrn1.i, 0, 1
  ret %struct.float32x2x2_t %.fca.0.1.insert
}

define %struct.poly8x8x2_t @test_vtrn_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtrn_p8:
; CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vtrn.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %vtrn1.i = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.poly8x8x2_t undef, <8 x i8> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly8x8x2_t %.fca.0.0.insert, <8 x i8> %vtrn1.i, 0, 1
  ret %struct.poly8x8x2_t %.fca.0.1.insert
}

define %struct.poly16x4x2_t @test_vtrn_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vtrn_p16:
; CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vtrn.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %vtrn1.i = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.poly16x4x2_t undef, <4 x i16> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly16x4x2_t %.fca.0.0.insert, <4 x i16> %vtrn1.i, 0, 1
  ret %struct.poly16x4x2_t %.fca.0.1.insert
}

define %struct.int8x16x2_t @test_vtrnq_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrnq_s8:
; CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vtrn.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  %vtrn1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  %.fca.0.0.insert = insertvalue %struct.int8x16x2_t undef, <16 x i8> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x2_t %.fca.0.0.insert, <16 x i8> %vtrn1.i, 0, 1
  ret %struct.int8x16x2_t %.fca.0.1.insert
}

define %struct.int16x8x2_t @test_vtrnq_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrnq_s16:
; CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vtrn.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %vtrn1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.int16x8x2_t undef, <8 x i16> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x2_t %.fca.0.0.insert, <8 x i16> %vtrn1.i, 0, 1
  ret %struct.int16x8x2_t %.fca.0.1.insert
}

define %struct.int32x4x2_t @test_vtrnq_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vtrnq_s32:
; CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vtrn.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %vtrn1.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.int32x4x2_t undef, <4 x i32> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x2_t %.fca.0.0.insert, <4 x i32> %vtrn1.i, 0, 1
  ret %struct.int32x4x2_t %.fca.0.1.insert
}

define %struct.uint8x16x2_t @test_vtrnq_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrnq_u8:
; CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vtrn.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  %vtrn1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  %.fca.0.0.insert = insertvalue %struct.uint8x16x2_t undef, <16 x i8> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint8x16x2_t %.fca.0.0.insert, <16 x i8> %vtrn1.i, 0, 1
  ret %struct.uint8x16x2_t %.fca.0.1.insert
}

define %struct.uint16x8x2_t @test_vtrnq_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrnq_u16:
; CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vtrn.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %vtrn1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.uint16x8x2_t undef, <8 x i16> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint16x8x2_t %.fca.0.0.insert, <8 x i16> %vtrn1.i, 0, 1
  ret %struct.uint16x8x2_t %.fca.0.1.insert
}

define %struct.uint32x4x2_t @test_vtrnq_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vtrnq_u32:
; CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vtrn.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %vtrn1.i = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.uint32x4x2_t undef, <4 x i32> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint32x4x2_t %.fca.0.0.insert, <4 x i32> %vtrn1.i, 0, 1
  ret %struct.uint32x4x2_t %.fca.0.1.insert
}

define %struct.float32x4x2_t @test_vtrnq_f32(<4 x float> %a, <4 x float> %b) {
; CHECK: test_vtrnq_f32:
; CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vtrn.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %vtrn1.i = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.float32x4x2_t undef, <4 x float> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x2_t %.fca.0.0.insert, <4 x float> %vtrn1.i, 0, 1
  ret %struct.float32x4x2_t %.fca.0.1.insert
}

define %struct.poly8x16x2_t @test_vtrnq_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vtrnq_p8:
; CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vtrn.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  %vtrn1.i = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  %.fca.0.0.insert = insertvalue %struct.poly8x16x2_t undef, <16 x i8> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly8x16x2_t %.fca.0.0.insert, <16 x i8> %vtrn1.i, 0, 1
  ret %struct.poly8x16x2_t %.fca.0.1.insert
}

define %struct.poly16x8x2_t @test_vtrnq_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vtrnq_p16:
; CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vtrn.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %vtrn1.i = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.poly16x8x2_t undef, <8 x i16> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.poly16x8x2_t %.fca.0.0.insert, <8 x i16> %vtrn1.i, 0, 1
  ret %struct.poly16x8x2_t %.fca.0.1.insert
}

define %struct.uint8x8x2_t @test_uzp(<16 x i8> %y) {
; CHECK: test_uzp:

  %vuzp.i = shufflevector <16 x i8> %y, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vuzp1.i = shufflevector <16 x i8> %y, <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %.fca.0.0.insert = insertvalue %struct.uint8x8x2_t undef, <8 x i8> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.uint8x8x2_t %.fca.0.0.insert, <8 x i8> %vuzp1.i, 0, 1
  ret %struct.uint8x8x2_t %.fca.0.1.insert

; CHECK: dup	{{d[0-9]+}}, {{v[0-9]+}}.d[1]
; CHECK-NEXT: uzp1	{{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK-NEXT: uzp2	{{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
