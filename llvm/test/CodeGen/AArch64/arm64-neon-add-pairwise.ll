; RUN: llc -mtriple=arm64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=arm64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_addp_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: test_addp_v8i8:
  %tmp1 = call <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: addp v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

declare <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_addp_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_addp_v16i8:
  %tmp1 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: addp v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

declare <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_addp_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_addp_v4i16:
  %tmp1 = call <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: addp v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

declare <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_addp_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_addp_v8i16:
  %tmp1 = call <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: addp v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

declare <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_addp_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_addp_v2i32:
  %tmp1 = call <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: addp v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

declare <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_addp_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_addp_v4i32:
  %tmp1 = call <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: addp v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}


declare <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64>, <2 x i64>)

define <2 x i64> @test_addp_v2i64(<2 x i64> %lhs, <2 x i64> %rhs) {
; CHECK: test_addp_v2i64:
        %val = call <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64> %lhs, <2 x i64> %rhs)
; CHECK: addp v0.2d, v0.2d, v1.2d
        ret <2 x i64> %val
}

declare <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.aarch64.neon.faddp.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.faddp.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_faddp_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_faddp_v2f32:
        %val = call <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: faddp v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_faddp_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_faddp_v4f32:
        %val = call <4 x float> @llvm.aarch64.neon.faddp.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: faddp v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_faddp_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_faddp_v2f64:
        %val = call <2 x double> @llvm.aarch64.neon.faddp.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: faddp v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}

define i32 @test_vaddv.v2i32(<2 x i32> %a) {
; CHECK-LABEL: test_vaddv.v2i32
; CHECK: addp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  %1 = tail call i32 @llvm.aarch64.neon.saddv.i32.v2i32(<2 x i32> %a)
  ret i32 %1
}

declare i32 @llvm.aarch64.neon.saddv.i32.v2i32(<2 x i32>)
