; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon -verify-machineinstrs < %s | FileCheck %s

; From <8 x i8>

define <1 x i64> @test_v8i8_to_v1i64(<8 x i8> %in) nounwind {
; CHECK: test_v8i8_to_v1i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i8> %in to <1 x i64>
  ret <1 x i64> %val
}

define <2 x i32> @test_v8i8_to_v2i32(<8 x i8> %in) nounwind {
; CHECK: test_v8i8_to_v2i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i8> %in to <2 x i32>
  ret <2 x i32> %val
}

define <2 x float> @test_v8i8_to_v2f32(<8 x i8> %in) nounwind{
; CHECK: test_v8i8_to_v2f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i8> %in to <2 x float>
  ret <2 x float> %val
}

define <4 x i16> @test_v8i8_to_v4i16(<8 x i8> %in) nounwind{
; CHECK: test_v8i8_to_v4i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i8> %in to <4 x i16>
  ret <4 x i16> %val
}

define <8 x i8> @test_v8i8_to_v8i8(<8 x i8> %in) nounwind{
; CHECK: test_v8i8_to_v8i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i8> %in to <8 x i8>
  ret <8 x i8> %val
}

; From <4 x i16>

define <1 x i64> @test_v4i16_to_v1i64(<4 x i16> %in) nounwind {
; CHECK: test_v4i16_to_v1i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i16> %in to <1 x i64>
  ret <1 x i64> %val
}

define <2 x i32> @test_v4i16_to_v2i32(<4 x i16> %in) nounwind {
; CHECK: test_v4i16_to_v2i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i16> %in to <2 x i32>
  ret <2 x i32> %val
}

define <2 x float> @test_v4i16_to_v2f32(<4 x i16> %in) nounwind{
; CHECK: test_v4i16_to_v2f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i16> %in to <2 x float>
  ret <2 x float> %val
}

define <4 x i16> @test_v4i16_to_v4i16(<4 x i16> %in) nounwind{
; CHECK: test_v4i16_to_v4i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i16> %in to <4 x i16>
  ret <4 x i16> %val
}

define <8 x i8> @test_v4i16_to_v8i8(<4 x i16> %in) nounwind{
; CHECK: test_v4i16_to_v8i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i16> %in to <8 x i8>
  ret <8 x i8> %val
}

; From <2 x i32>

define <1 x i64> @test_v2i32_to_v1i64(<2 x i32> %in) nounwind {
; CHECK: test_v2i32_to_v1i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i32> %in to <1 x i64>
  ret <1 x i64> %val
}

define <2 x i32> @test_v2i32_to_v2i32(<2 x i32> %in) nounwind {
; CHECK: test_v2i32_to_v2i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i32> %in to <2 x i32>
  ret <2 x i32> %val
}

define <2 x float> @test_v2i32_to_v2f32(<2 x i32> %in) nounwind{
; CHECK: test_v2i32_to_v2f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i32> %in to <2 x float>
  ret <2 x float> %val
}

define <4 x i16> @test_v2i32_to_v4i16(<2 x i32> %in) nounwind{
; CHECK: test_v2i32_to_v4i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i32> %in to <4 x i16>
  ret <4 x i16> %val
}

define <8 x i8> @test_v2i32_to_v8i8(<2 x i32> %in) nounwind{
; CHECK: test_v2i32_to_v8i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i32> %in to <8 x i8>
  ret <8 x i8> %val
}

; From <2 x float>

define <1 x i64> @test_v2f32_to_v1i64(<2 x float> %in) nounwind {
; CHECK: test_v2f32_to_v1i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x float> %in to <1 x i64>
  ret <1 x i64> %val
}

define <2 x i32> @test_v2f32_to_v2i32(<2 x float> %in) nounwind {
; CHECK: test_v2f32_to_v2i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x float> %in to <2 x i32>
  ret <2 x i32> %val
}

define <2 x float> @test_v2f32_to_v2f32(<2 x float> %in) nounwind{
; CHECK: test_v2f32_to_v2f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x float> %in to <2 x float>
  ret <2 x float> %val
}

define <4 x i16> @test_v2f32_to_v4i16(<2 x float> %in) nounwind{
; CHECK: test_v2f32_to_v4i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x float> %in to <4 x i16>
  ret <4 x i16> %val
}

define <8 x i8> @test_v2f32_to_v8i8(<2 x float> %in) nounwind{
; CHECK: test_v2f32_to_v8i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x float> %in to <8 x i8>
  ret <8 x i8> %val
}

; From <1 x i64>

define <1 x i64> @test_v1i64_to_v1i64(<1 x i64> %in) nounwind {
; CHECK: test_v1i64_to_v1i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <1 x i64> %in to <1 x i64>
  ret <1 x i64> %val
}

define <2 x i32> @test_v1i64_to_v2i32(<1 x i64> %in) nounwind {
; CHECK: test_v1i64_to_v2i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <1 x i64> %in to <2 x i32>
  ret <2 x i32> %val
}

define <2 x float> @test_v1i64_to_v2f32(<1 x i64> %in) nounwind{
; CHECK: test_v1i64_to_v2f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <1 x i64> %in to <2 x float>
  ret <2 x float> %val
}

define <4 x i16> @test_v1i64_to_v4i16(<1 x i64> %in) nounwind{
; CHECK: test_v1i64_to_v4i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <1 x i64> %in to <4 x i16>
  ret <4 x i16> %val
}

define <8 x i8> @test_v1i64_to_v8i8(<1 x i64> %in) nounwind{
; CHECK: test_v1i64_to_v8i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <1 x i64> %in to <8 x i8>
  ret <8 x i8> %val
}


; From <16 x i8>

define <2 x double> @test_v16i8_to_v2f64(<16 x i8> %in) nounwind {
; CHECK: test_v16i8_to_v2f64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <16 x i8> %in to <2 x double>
  ret <2 x double> %val
}

define <2 x i64> @test_v16i8_to_v2i64(<16 x i8> %in) nounwind {
; CHECK: test_v16i8_to_v2i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <16 x i8> %in to <2 x i64>
  ret <2 x i64> %val
}

define <4 x i32> @test_v16i8_to_v4i32(<16 x i8> %in) nounwind {
; CHECK: test_v16i8_to_v4i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <16 x i8> %in to <4 x i32>
  ret <4 x i32> %val
}

define <4 x float> @test_v16i8_to_v2f32(<16 x i8> %in) nounwind{
; CHECK: test_v16i8_to_v2f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <16 x i8> %in to <4 x float>
  ret <4 x float> %val
}

define <8 x i16> @test_v16i8_to_v8i16(<16 x i8> %in) nounwind{
; CHECK: test_v16i8_to_v8i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <16 x i8> %in to <8 x i16>
  ret <8 x i16> %val
}

define <16 x i8> @test_v16i8_to_v16i8(<16 x i8> %in) nounwind{
; CHECK: test_v16i8_to_v16i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <16 x i8> %in to <16 x i8>
  ret <16 x i8> %val
}

; From <8 x i16>

define <2 x double> @test_v8i16_to_v2f64(<8 x i16> %in) nounwind {
; CHECK: test_v8i16_to_v2f64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i16> %in to <2 x double>
  ret <2 x double> %val
}

define <2 x i64> @test_v8i16_to_v2i64(<8 x i16> %in) nounwind {
; CHECK: test_v8i16_to_v2i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i16> %in to <2 x i64>
  ret <2 x i64> %val
}

define <4 x i32> @test_v8i16_to_v4i32(<8 x i16> %in) nounwind {
; CHECK: test_v8i16_to_v4i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i16> %in to <4 x i32>
  ret <4 x i32> %val
}

define <4 x float> @test_v8i16_to_v2f32(<8 x i16> %in) nounwind{
; CHECK: test_v8i16_to_v2f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i16> %in to <4 x float>
  ret <4 x float> %val
}

define <8 x i16> @test_v8i16_to_v8i16(<8 x i16> %in) nounwind{
; CHECK: test_v8i16_to_v8i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i16> %in to <8 x i16>
  ret <8 x i16> %val
}

define <16 x i8> @test_v8i16_to_v16i8(<8 x i16> %in) nounwind{
; CHECK: test_v8i16_to_v16i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <8 x i16> %in to <16 x i8>
  ret <16 x i8> %val
}

; From <4 x i32>

define <2 x double> @test_v4i32_to_v2f64(<4 x i32> %in) nounwind {
; CHECK: test_v4i32_to_v2f64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i32> %in to <2 x double>
  ret <2 x double> %val
}

define <2 x i64> @test_v4i32_to_v2i64(<4 x i32> %in) nounwind {
; CHECK: test_v4i32_to_v2i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i32> %in to <2 x i64>
  ret <2 x i64> %val
}

define <4 x i32> @test_v4i32_to_v4i32(<4 x i32> %in) nounwind {
; CHECK: test_v4i32_to_v4i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i32> %in to <4 x i32>
  ret <4 x i32> %val
}

define <4 x float> @test_v4i32_to_v2f32(<4 x i32> %in) nounwind{
; CHECK: test_v4i32_to_v2f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i32> %in to <4 x float>
  ret <4 x float> %val
}

define <8 x i16> @test_v4i32_to_v8i16(<4 x i32> %in) nounwind{
; CHECK: test_v4i32_to_v8i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i32> %in to <8 x i16>
  ret <8 x i16> %val
}

define <16 x i8> @test_v4i32_to_v16i8(<4 x i32> %in) nounwind{
; CHECK: test_v4i32_to_v16i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x i32> %in to <16 x i8>
  ret <16 x i8> %val
}

; From <4 x float>

define <2 x double> @test_v4f32_to_v2f64(<4 x float> %in) nounwind {
; CHECK: test_v4f32_to_v2f64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x float> %in to <2 x double>
  ret <2 x double> %val
}

define <2 x i64> @test_v4f32_to_v2i64(<4 x float> %in) nounwind {
; CHECK: test_v4f32_to_v2i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x float> %in to <2 x i64>
  ret <2 x i64> %val
}

define <4 x i32> @test_v4f32_to_v4i32(<4 x float> %in) nounwind {
; CHECK: test_v4f32_to_v4i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x float> %in to <4 x i32>
  ret <4 x i32> %val
}

define <4 x float> @test_v4f32_to_v4f32(<4 x float> %in) nounwind{
; CHECK: test_v4f32_to_v4f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x float> %in to <4 x float>
  ret <4 x float> %val
}

define <8 x i16> @test_v4f32_to_v8i16(<4 x float> %in) nounwind{
; CHECK: test_v4f32_to_v8i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x float> %in to <8 x i16>
  ret <8 x i16> %val
}

define <16 x i8> @test_v4f32_to_v16i8(<4 x float> %in) nounwind{
; CHECK: test_v4f32_to_v16i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <4 x float> %in to <16 x i8>
  ret <16 x i8> %val
}

; From <2 x i64>

define <2 x double> @test_v2i64_to_v2f64(<2 x i64> %in) nounwind {
; CHECK: test_v2i64_to_v2f64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i64> %in to <2 x double>
  ret <2 x double> %val
}

define <2 x i64> @test_v2i64_to_v2i64(<2 x i64> %in) nounwind {
; CHECK: test_v2i64_to_v2i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i64> %in to <2 x i64>
  ret <2 x i64> %val
}

define <4 x i32> @test_v2i64_to_v4i32(<2 x i64> %in) nounwind {
; CHECK: test_v2i64_to_v4i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i64> %in to <4 x i32>
  ret <4 x i32> %val
}

define <4 x float> @test_v2i64_to_v4f32(<2 x i64> %in) nounwind{
; CHECK: test_v2i64_to_v4f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i64> %in to <4 x float>
  ret <4 x float> %val
}

define <8 x i16> @test_v2i64_to_v8i16(<2 x i64> %in) nounwind{
; CHECK: test_v2i64_to_v8i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i64> %in to <8 x i16>
  ret <8 x i16> %val
}

define <16 x i8> @test_v2i64_to_v16i8(<2 x i64> %in) nounwind{
; CHECK: test_v2i64_to_v16i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x i64> %in to <16 x i8>
  ret <16 x i8> %val
}

; From <2 x double>

define <2 x double> @test_v2f64_to_v2f64(<2 x double> %in) nounwind {
; CHECK: test_v2f64_to_v2f64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x double> %in to <2 x double>
  ret <2 x double> %val
}

define <2 x i64> @test_v2f64_to_v2i64(<2 x double> %in) nounwind {
; CHECK: test_v2f64_to_v2i64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x double> %in to <2 x i64>
  ret <2 x i64> %val
}

define <4 x i32> @test_v2f64_to_v4i32(<2 x double> %in) nounwind {
; CHECK: test_v2f64_to_v4i32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x double> %in to <4 x i32>
  ret <4 x i32> %val
}

define <4 x float> @test_v2f64_to_v4f32(<2 x double> %in) nounwind{
; CHECK: test_v2f64_to_v4f32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x double> %in to <4 x float>
  ret <4 x float> %val
}

define <8 x i16> @test_v2f64_to_v8i16(<2 x double> %in) nounwind{
; CHECK: test_v2f64_to_v8i16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x double> %in to <8 x i16>
  ret <8 x i16> %val
}

define <16 x i8> @test_v2f64_to_v16i8(<2 x double> %in) nounwind{
; CHECK: test_v2f64_to_v16i8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: ret

  %val = bitcast <2 x double> %in to <16 x i8>
  ret <16 x i8> %val
}

