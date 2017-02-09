; RUN: llc < %s -mtriple aarch64-unknown-unknown -aarch64-neon-syntax=apple -asm-verbose=false | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Test the (concat_vectors (bitcast (scalar)), ..) pattern.

define <8 x i8> @test_concat_scalar_v2i8_to_v8i8_dup(i32 %x) #0 {
entry:
; CHECK-LABEL: test_concat_scalar_v2i8_to_v8i8_dup:
; CHECK-NEXT: dup.4h v0, w0
; CHECK-NEXT: ret
  %t = trunc i32 %x to i16
  %0 = bitcast i16 %t to <2 x i8>
  %1 = shufflevector <2 x i8> %0, <2 x i8> undef, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  ret <8 x i8> %1
}

define <8 x i8> @test_concat_scalar_v4i8_to_v8i8_dup(i32 %x) #0 {
entry:
; CHECK-LABEL: test_concat_scalar_v4i8_to_v8i8_dup:
; CHECK-NEXT: dup.2s v0, w0
; CHECK-NEXT: ret
  %0 = bitcast i32 %x to <4 x i8>
  %1 = shufflevector <4 x i8> %0, <4 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  ret <8 x i8> %1
}

define <8 x i16> @test_concat_scalar_v2i16_to_v8i16_dup(i32 %x) #0 {
entry:
; CHECK-LABEL: test_concat_scalar_v2i16_to_v8i16_dup:
; CHECK-NEXT: dup.4s v0, w0
; CHECK-NEXT: ret
  %0 = bitcast i32 %x to <2 x i16>
  %1 = shufflevector <2 x i16> %0, <2 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 2, i32 0, i32 1, i32 0, i32 1>
  ret <8 x i16> %1
}

define <8 x i8> @test_concat_scalars_2x_v2i8_to_v8i8(i32 %x, i32 %y) #0 {
entry:
; CHECK-LABEL: test_concat_scalars_2x_v2i8_to_v8i8:
; CHECK-NEXT: ins.h v0[0], w0
; CHECK-NEXT: ins.h v0[1], w1
; CHECK-NEXT: ins.h v0[3], w1
; CHECK-NEXT: ret
  %tx = trunc i32 %x to i16
  %ty = trunc i32 %y to i16
  %bx = bitcast i16 %tx to <2 x i8>
  %by = bitcast i16 %ty to <2 x i8>
  %r = shufflevector <2 x i8> %bx, <2 x i8> %by, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 2, i32 3>
  ret <8 x i8> %r
}

define <8 x i8> @test_concat_scalars_2x_v4i8_to_v8i8_dup(i32 %x, i32 %y) #0 {
entry:
; CHECK-LABEL: test_concat_scalars_2x_v4i8_to_v8i8_dup:
; CHECK-NEXT: fmov s0, w1
; CHECK-NEXT: ins.s v0[1], w0
; CHECK-NEXT: ret
  %bx = bitcast i32 %x to <4 x i8>
  %by = bitcast i32 %y to <4 x i8>
  %r = shufflevector <4 x i8> %bx, <4 x i8> %by, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <8 x i8> %r
}

define <8 x i16> @test_concat_scalars_2x_v2i16_to_v8i16_dup(i32 %x, i32 %y) #0 {
entry:
; CHECK-LABEL: test_concat_scalars_2x_v2i16_to_v8i16_dup:
; CHECK-NEXT: fmov s0, w0
; CHECK-NEXT: ins.s v0[1], w1
; CHECK-NEXT: ins.s v0[2], w1
; CHECK-NEXT: ins.s v0[3], w0
; CHECK-NEXT: ret
  %bx = bitcast i32 %x to <2 x i16>
  %by = bitcast i32 %y to <2 x i16>
  %r = shufflevector <2 x i16> %bx, <2 x i16> %by, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 0, i32 1>
  ret <8 x i16> %r
}

; Also make sure we minimize bitcasts.

; This is a pretty artificial testcase: make sure we bitcast to floating-point
; if any of the scalars is floating-point.
define <8 x i8> @test_concat_scalars_mixed_2x_v2i8_to_v8i8(float %dummy, i32 %x, half %y) #0 {
entry:
; CHECK-LABEL: test_concat_scalars_mixed_2x_v2i8_to_v8i8:
; CHECK-NEXT: fmov s[[X:[0-9]+]], w0
; CHECK-NEXT: ins.h v0[0], v[[X]][0]
; CHECK-NEXT: ins.h v0[1], v1[0]
; CHECK-NEXT: ins.h v0[2], v[[X]][0]
; CHECK-NEXT: ins.h v0[3], v1[0]
; CHECK-NEXT: ret
  %t = trunc i32 %x to i16
  %0 = bitcast i16 %t to <2 x i8>
  %y0 = bitcast half %y to <2 x i8>
  %1 = shufflevector <2 x i8> %0, <2 x i8> %y0, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  ret <8 x i8> %1
}

define <2 x float> @test_concat_scalars_fp_2x_v2i8_to_v8i8(float %dummy, half %x, half %y) #0 {
entry:
; CHECK-LABEL: test_concat_scalars_fp_2x_v2i8_to_v8i8:
; CHECK-NEXT: ins.h v0[0], v1[0]
; CHECK-NEXT: ins.h v0[1], v2[0]
; CHECK-NEXT: ins.h v0[2], v1[0]
; CHECK-NEXT: ins.h v0[3], v2[0]
; CHECK-NEXT: ret
  %0 = bitcast half %x to <2 x i8>
  %y0 = bitcast half %y to <2 x i8>
  %1 = shufflevector <2 x i8> %0, <2 x i8> %y0, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  %2 = bitcast <8 x i8> %1 to <2 x float>
  ret <2 x float> %2
}

define <4 x float> @test_concat_scalar_fp_v2i16_to_v16i8_dup(float %x) #0 {
entry:
; CHECK-LABEL: test_concat_scalar_fp_v2i16_to_v16i8_dup:
; CHECK-NEXT: dup.4s v0, v0[0]
; CHECK-NEXT: ret
  %0 = bitcast float %x to <2 x i16>
  %1 = shufflevector <2 x i16> %0, <2 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 2, i32 0, i32 1, i32 0, i32 1>
  %2 = bitcast <8 x i16> %1 to <4 x float>
  ret <4 x float> %2
}

attributes #0 = { nounwind }
