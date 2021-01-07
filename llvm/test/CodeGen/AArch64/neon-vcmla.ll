; RUN: llc %s -mtriple=aarch64 -mattr=+v8.3a,+fullfp16 -o - | FileCheck %s

define <4 x half> @test_16x4(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
entry:
; CHECK-LABEL: test_16x4
; CHECK: fcmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, #0
;
  %res = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot0.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  ret <4 x half> %res
}

define <4 x half> @test_16x4_lane_1(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
entry:
; CHECK-LABEL: test_16x4_lane_1
; CHECK: fcmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[1], #0
;
  %c.cast = bitcast <4 x half> %c to <2 x i32>
  %c.dup = shufflevector <2 x i32> %c.cast , <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %c.res = bitcast <2 x i32> %c.dup to <4 x half>
  %res = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot0.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c.res)
  ret <4 x half> %res
}

define <4 x half> @test_rot90_16x4(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
entry:
; CHECK-LABEL: test_rot90_16x4
; CHECK: fcmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, #90
;
  %res = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot90.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  ret <4 x half> %res
}

define <4 x half> @test_rot90_16x4_lane_0(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
entry:
; CHECK-LABEL: test_rot90_16x4_lane_0
; CHECK: fcmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[0], #90
;
  %c.cast = bitcast <4 x half> %c to <2 x i32>
  %c.dup = shufflevector <2 x i32> %c.cast , <2 x i32> undef, <2 x i32> <i32 0, i32 0>
  %c.res = bitcast <2 x i32> %c.dup to <4 x half>
  %res = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot90.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c.res)
  ret <4 x half> %res
}

define <4 x half> @test_rot180_16x4(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
entry:
; CHECK-LABEL: test_rot180_16x4
; CHECK: fcmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, #180
;
  %res = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot180.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  ret <4 x half> %res
}

define <4 x half> @test_rot180_16x4_lane_0(<4 x half> %a, <4 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_rot180_16x4_lane_0
; CHECK: fcmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[0], #180

  %c.cast = bitcast <8 x half> %c to <4 x i32>
  %c.dup = shufflevector <4 x i32> %c.cast , <4 x i32> undef, <2 x i32> <i32 0, i32 0>
  %c.res = bitcast <2 x i32> %c.dup to <4 x half>
  %res = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot180.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c.res)
  ret <4 x half> %res
}

define <4 x half> @test_rot270_16x4(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
entry:
; CHECK-LABEL: test_rot270_16x4
; CHECK: fcmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, #270
;
  %res = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot270.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  ret <4 x half> %res
}

define <2 x float> @test_32x2(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
entry:
; CHECK-LABEL: test_32x2
; CHECK: fcmla v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, #0
;
  %res = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot0.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c)
  ret <2 x float> %res
}

define <2 x float> @test_rot90_32x2(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
entry:
; CHECK-LABEL: test_rot90_32x2
; CHECK: fcmla v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, #90
;
  %res = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot90.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c)
  ret <2 x float> %res
}

define <2 x float> @test_rot180_32x2(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
entry:
; CHECK-LABEL: test_rot180_32x2
; CHECK: fcmla v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, #180
;
  %res = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot180.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c)
  ret <2 x float> %res
}

define <2 x float> @test_rot270_32x2(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
entry:
; CHECK-LABEL: test_rot270_32x2
; CHECK: fcmla v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, #270
;
  %res = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot270.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c)
  ret <2 x float> %res
}

define <8 x half> @test_16x8(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_16x8
; CHECK: fcmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, #0
;
  %res = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot0.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c)
  ret <8 x half> %res
}

define <8 x half> @test_16x8_lane_0(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_16x8_lane_0
; CHECK: fcmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[0], #0
;
  %c.cast = bitcast <8 x half> %c to <4 x i32>
  %c.dup = shufflevector <4 x i32> %c.cast , <4 x i32> undef, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %c.res = bitcast <4 x i32> %c.dup to <8 x half>
  %res = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot0.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c.res)
  ret <8 x half> %res
}

define <8 x half> @test_rot90_16x8(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_rot90_16x8
; CHECK: fcmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, #90
;
  %res = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot90.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c)
  ret <8 x half> %res
}

define <8 x half> @test_rot90_16x8_lane_1(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_rot90_16x8_lane_1
; CHECK: fcmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[1], #90
;
  %c.cast = bitcast <8 x half> %c to <4 x i32>
  %c.dup = shufflevector <4 x i32> %c.cast , <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %c.res = bitcast <4 x i32> %c.dup to <8 x half>
  %res = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot90.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c.res)
  ret <8 x half> %res
}

define <8 x half> @test_rot180_16x8(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_rot180_16x8
; CHECK: fcmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, #180
;
  %res = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot180.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c)
  ret <8 x half> %res
}

define <8 x half> @test_rot180_16x8_lane_1(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_rot180_16x8_lane_1
; CHECK: fcmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[1], #180
;
  %c.cast = bitcast <8 x half> %c to <4 x i32>
  %c.dup = shufflevector <4 x i32> %c.cast , <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %c.res = bitcast <4 x i32> %c.dup to <8 x half>
  %res = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot180.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c.res)
  ret <8 x half> %res
}

define <8 x half> @test_rot270_16x8(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_rot270_16x8
; CHECK: fcmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, #270
;
  %res = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot270.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c)
  ret <8 x half> %res
}

define <8 x half> @test_rot270_16x8_lane_0(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
entry:
; CHECK-LABEL: test_rot270_16x8_lane_0
; CHECK: fcmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[0], #270
;
  %c.cast = bitcast <8 x half> %c to <4 x i32>
  %c.dup = shufflevector <4 x i32> %c.cast , <4 x i32> undef, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %c.res = bitcast <4 x i32> %c.dup to <8 x half>
  %res = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot270.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c.res)
  ret <8 x half> %res
}

define <4 x float> @test_32x4(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
; CHECK-LABEL: test_32x4
; CHECK: fcmla v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, #0
;
  %res = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot0.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %res
}

define <4 x float> @test_32x4_lane_0(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
; CHECK-LABEL: test_32x4_lane_0
; CHECK: fcmla v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.s[0], #0
;
  %c.cast = bitcast <4 x float> %c to <2 x i64>
  %c.dup = shufflevector <2 x i64> %c.cast , <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %c.res = bitcast <2 x i64> %c.dup to <4 x float>
  %res = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot0.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c.res)
  ret <4 x float> %res
}

define <4 x float> @test_rot90_32x4(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
; CHECK-LABEL: test_rot90_32x4
; CHECK: fcmla v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, #90
;
  %res = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot90.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %res
}

define <4 x float> @test_rot180_32x4(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
; CHECK-LABEL: test_rot180_32x4
; CHECK: fcmla v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, #180
;
  %res = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot180.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %res
}

define <4 x float> @test_rot270_32x4(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
; CHECK-LABEL: test_rot270_32x4
; CHECK: fcmla v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, #270
;
  %res = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot270.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %res
}

define <2 x double> @test_64x2(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
; CHECK-LABEL: test_64x2
; CHECK: fcmla v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, #0
;
  %res = tail call <2 x double> @llvm.aarch64.neon.vcmla.rot0.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
  ret <2 x double> %res
}

define <2 x double> @test_rot90_64x2(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
; CHECK-LABEL: test_rot90_64x2
; CHECK: fcmla v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, #90
;
  %res = tail call <2 x double> @llvm.aarch64.neon.vcmla.rot90.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
  ret <2 x double> %res
}

define <2 x double> @test_rot180_64x2(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
; CHECK-LABEL: test_rot180_64x2
; CHECK: fcmla v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, #180
;
  %res = tail call <2 x double> @llvm.aarch64.neon.vcmla.rot180.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
  ret <2 x double> %res
}

define <2 x double> @test_rot270_64x2(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
; CHECK-LABEL: test_rot270_64x2
; CHECK: fcmla v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, #270
;
  %res = tail call <2 x double> @llvm.aarch64.neon.vcmla.rot270.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
  ret <2 x double> %res
}

declare <4 x half> @llvm.aarch64.neon.vcmla.rot0.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <4 x half> @llvm.aarch64.neon.vcmla.rot90.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <4 x half> @llvm.aarch64.neon.vcmla.rot180.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <4 x half> @llvm.aarch64.neon.vcmla.rot270.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <8 x half> @llvm.aarch64.neon.vcmla.rot0.v8f16(<8 x half>, <8 x half>, <8 x half>)
declare <8 x half> @llvm.aarch64.neon.vcmla.rot90.v8f16(<8 x half>, <8 x half>, <8 x half>)
declare <8 x half> @llvm.aarch64.neon.vcmla.rot180.v8f16(<8 x half>, <8 x half>, <8 x half>)
declare <8 x half> @llvm.aarch64.neon.vcmla.rot270.v8f16(<8 x half>, <8 x half>, <8 x half>)
declare <2 x float> @llvm.aarch64.neon.vcmla.rot0.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <2 x float> @llvm.aarch64.neon.vcmla.rot90.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <2 x float> @llvm.aarch64.neon.vcmla.rot180.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <2 x float> @llvm.aarch64.neon.vcmla.rot270.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <4 x float> @llvm.aarch64.neon.vcmla.rot0.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.aarch64.neon.vcmla.rot90.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.aarch64.neon.vcmla.rot180.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.aarch64.neon.vcmla.rot270.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.vcmla.rot0.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare <2 x double> @llvm.aarch64.neon.vcmla.rot90.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare <2 x double> @llvm.aarch64.neon.vcmla.rot180.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare <2 x double> @llvm.aarch64.neon.vcmla.rot270.v2f64(<2 x double>, <2 x double>, <2 x double>)
