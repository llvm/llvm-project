; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <2 x i64> @test_v2f32_to_signed_v2i64(<2 x float> %in) {
; CHECK-LABEL: test_v2f32_to_signed_v2i64:
; CHECK: fcvtl [[VAL64:v[0-9]+]].2d, v0.2s
; CHECK: fcvtzs.2d v0, [[VAL64]]

  %val = fptosi <2 x float> %in to <2 x i64>
  ret <2 x i64> %val
}

define <2 x i64> @test_v2f32_to_unsigned_v2i64(<2 x float> %in) {
; CHECK-LABEL: test_v2f32_to_unsigned_v2i64:
; CHECK: fcvtl [[VAL64:v[0-9]+]].2d, v0.2s
; CHECK: fcvtzu.2d v0, [[VAL64]]

  %val = fptoui <2 x float> %in to <2 x i64>
  ret <2 x i64> %val
}

define <2 x i16> @test_v2f32_to_signed_v2i16(<2 x float> %in) {
; CHECK-LABEL: test_v2f32_to_signed_v2i16:
; CHECK: fcvtzs.2s v0, v0

  %val = fptosi <2 x float> %in to <2 x i16>
  ret <2 x i16> %val
}

define <2 x i16> @test_v2f32_to_unsigned_v2i16(<2 x float> %in) {
; CHECK-LABEL: test_v2f32_to_unsigned_v2i16:
; CHECK: fcvtzs.2s v0, v0

  %val = fptoui <2 x float> %in to <2 x i16>
  ret <2 x i16> %val
}

define <2 x i8> @test_v2f32_to_signed_v2i8(<2 x float> %in) {
; CHECK-LABEL: test_v2f32_to_signed_v2i8:
; CHECK: fcvtzs.2s v0, v0

  %val = fptosi <2 x float> %in to <2 x i8>
  ret <2 x i8> %val
}

define <2 x i8> @test_v2f32_to_unsigned_v2i8(<2 x float> %in) {
; CHECK-LABEL: test_v2f32_to_unsigned_v2i8:
; CHECK: fcvtzs.2s v0, v0

  %val = fptoui <2 x float> %in to <2 x i8>
  ret <2 x i8> %val
}

define <4 x i16> @test_v4f32_to_signed_v4i16(<4 x float> %in) {
; CHECK-LABEL: test_v4f32_to_signed_v4i16:
; CHECK: fcvtzs.4s [[VAL64:v[0-9]+]], v0
; CHECK: xtn.4h v0, [[VAL64]]

  %val = fptosi <4 x float> %in to <4 x i16>
  ret <4 x i16> %val
}

define <4 x i16> @test_v4f32_to_unsigned_v4i16(<4 x float> %in) {
; CHECK-LABEL: test_v4f32_to_unsigned_v4i16:
; CHECK: fcvtzu.4s [[VAL64:v[0-9]+]], v0
; CHECK: xtn.4h v0, [[VAL64]]

  %val = fptoui <4 x float> %in to <4 x i16>
  ret <4 x i16> %val
}

define <4 x i8> @test_v4f32_to_signed_v4i8(<4 x float> %in) {
; CHECK-LABEL: test_v4f32_to_signed_v4i8:
; CHECK: fcvtzs.4s [[VAL64:v[0-9]+]], v0
; CHECK: xtn.4h v0, [[VAL64]]

  %val = fptosi <4 x float> %in to <4 x i8>
  ret <4 x i8> %val
}

define <4 x i8> @test_v4f32_to_unsigned_v4i8(<4 x float> %in) {
; CHECK-LABEL: test_v4f32_to_unsigned_v4i8:
; CHECK: fcvtzs.4s [[VAL64:v[0-9]+]], v0
; CHECK: xtn.4h v0, [[VAL64]]

  %val = fptoui <4 x float> %in to <4 x i8>
  ret <4 x i8> %val
}

define <2 x i32> @test_v2f64_to_signed_v2i32(<2 x double> %in) {
; CHECK-LABEL: test_v2f64_to_signed_v2i32:
; CHECK: fcvtzs.2d [[VAL64:v[0-9]+]], v0
; CHECK: xtn.2s v0, [[VAL64]]

  %val = fptosi <2 x double> %in to <2 x i32>
  ret <2 x i32> %val
}

define <2 x i32> @test_v2f64_to_unsigned_v2i32(<2 x double> %in) {
; CHECK-LABEL: test_v2f64_to_unsigned_v2i32:
; CHECK: fcvtzu.2d [[VAL64:v[0-9]+]], v0
; CHECK: xtn.2s v0, [[VAL64]]

  %val = fptoui <2 x double> %in to <2 x i32>
  ret <2 x i32> %val
}

define <2 x i16> @test_v2f64_to_signed_v2i16(<2 x double> %in) {
; CHECK-LABEL: test_v2f64_to_signed_v2i16:
; CHECK: fcvtzs.2d [[VAL64:v[0-9]+]], v0
; CHECK: xtn.2s v0, [[VAL64]]

  %val = fptosi <2 x double> %in to <2 x i16>
  ret <2 x i16> %val
}

define <2 x i16> @test_v2f64_to_unsigned_v2i16(<2 x double> %in) {
; CHECK-LABEL: test_v2f64_to_unsigned_v2i16:
; CHECK: fcvtzs.2d [[VAL64:v[0-9]+]], v0
; CHECK: xtn.2s v0, [[VAL64]]

  %val = fptoui <2 x double> %in to <2 x i16>
  ret <2 x i16> %val
}

define <2 x i8> @test_v2f64_to_signed_v2i8(<2 x double> %in) {
; CHECK-LABEL: test_v2f64_to_signed_v2i8:
; CHECK: fcvtzs.2d [[VAL64:v[0-9]+]], v0
; CHECK: xtn.2s v0, [[VAL64]]

  %val = fptosi <2 x double> %in to <2 x i8>
  ret <2 x i8> %val
}

define <2 x i8> @test_v2f64_to_unsigned_v2i8(<2 x double> %in) {
; CHECK-LABEL: test_v2f64_to_unsigned_v2i8:
; CHECK: fcvtzs.2d [[VAL64:v[0-9]+]], v0
; CHECK: xtn.2s v0, [[VAL64]]

  %val = fptoui <2 x double> %in to <2 x i8>
  ret <2 x i8> %val
}
