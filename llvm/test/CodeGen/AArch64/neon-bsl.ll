; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

declare <2 x double> @llvm.arm.neon.vbsl.v2f64(<2 x double>, <2 x double>, <2 x double>)

declare <8 x i16> @llvm.arm.neon.vbsl.v8i16(<8 x i16>, <8 x i16>, <8 x i16>)

declare <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)

declare <4 x float> @llvm.arm.neon.vbsl.v4f32(<4 x float>, <4 x float>, <4 x float>)

declare <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)

declare <4 x i32> @llvm.arm.neon.vbsl.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)

declare <4 x i16> @llvm.arm.neon.vbsl.v4i16(<4 x i16>, <4 x i16>, <4 x i16>)

declare <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8>, <8 x i8>, <8 x i8>)

declare <1 x double> @llvm.arm.neon.vbsl.v1f64(<1 x double>, <1 x double>, <1 x double>)

declare <2 x float> @llvm.arm.neon.vbsl.v2f32(<2 x float>, <2 x float>, <2 x float>)

declare <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64>, <1 x i64>, <1 x i64>)

declare <2 x i32> @llvm.arm.neon.vbsl.v2i32(<2 x i32>, <2 x i32>, <2 x i32>)

define <8 x i8> @test_vbsl_s8(<8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3) {
; CHECK-LABEL: test_vbsl_s8:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3)
  ret <8 x i8> %vbsl.i
}

define <8 x i8> @test_vbsl_s16(<4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3) {
; CHECK-LABEL: test_vbsl_s16:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl3.i = tail call <4 x i16> @llvm.arm.neon.vbsl.v4i16(<4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3)
  %0 = bitcast <4 x i16> %vbsl3.i to <8 x i8>
  ret <8 x i8> %0
}

define <2 x i32> @test_vbsl_s32(<2 x i32> %v1, <2 x i32> %v2, <2 x i32> %v3) {
; CHECK-LABEL: test_vbsl_s32:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl3.i = tail call <2 x i32> @llvm.arm.neon.vbsl.v2i32(<2 x i32> %v1, <2 x i32> %v2, <2 x i32> %v3)
  ret <2 x i32> %vbsl3.i
}

define <1 x i64> @test_vbsl_s64(<1 x i64> %v1, <1 x i64> %v2, <1 x i64> %v3) {
; CHECK-LABEL: test_vbsl_s64:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl3.i = tail call <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64> %v1, <1 x i64> %v2, <1 x i64> %v3)
  ret <1 x i64> %vbsl3.i
}

define <8 x i8> @test_vbsl_u8(<8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3) {
; CHECK-LABEL: test_vbsl_u8:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3)
  ret <8 x i8> %vbsl.i
}

define <4 x i16> @test_vbsl_u16(<4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3) {
; CHECK-LABEL: test_vbsl_u16:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl3.i = tail call <4 x i16> @llvm.arm.neon.vbsl.v4i16(<4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3)
  ret <4 x i16> %vbsl3.i
}

define <2 x i32> @test_vbsl_u32(<2 x i32> %v1, <2 x i32> %v2, <2 x i32> %v3) {
; CHECK-LABEL: test_vbsl_u32:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl3.i = tail call <2 x i32> @llvm.arm.neon.vbsl.v2i32(<2 x i32> %v1, <2 x i32> %v2, <2 x i32> %v3)
  ret <2 x i32> %vbsl3.i
}

define <1 x i64> @test_vbsl_u64(<1 x i64> %v1, <1 x i64> %v2, <1 x i64> %v3) {
; CHECK-LABEL: test_vbsl_u64:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl3.i = tail call <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64> %v1, <1 x i64> %v2, <1 x i64> %v3)
  ret <1 x i64> %vbsl3.i
}

define <2 x float> @test_vbsl_f32(<2 x float> %v1, <2 x float> %v2, <2 x float> %v3) {
; CHECK-LABEL: test_vbsl_f32:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl3.i = tail call <2 x float> @llvm.arm.neon.vbsl.v2f32(<2 x float> %v1, <2 x float> %v2, <2 x float> %v3)
  ret <2 x float> %vbsl3.i
}

define <1 x double> @test_vbsl_f64(<1 x i64> %v1, <1 x double> %v2, <1 x double> %v3) {
; CHECK-LABEL: test_vbsl_f64:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl.i = bitcast <1 x i64> %v1 to <1 x double>
  %vbsl3.i = tail call <1 x double> @llvm.arm.neon.vbsl.v1f64(<1 x double> %vbsl.i, <1 x double> %v2, <1 x double> %v3)
  ret <1 x double> %vbsl3.i
}

define <8 x i8> @test_vbsl_p8(<8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3) {
; CHECK-LABEL: test_vbsl_p8:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %v1, <8 x i8> %v2, <8 x i8> %v3)
  ret <8 x i8> %vbsl.i
}

define <4 x i16> @test_vbsl_p16(<4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3) {
; CHECK-LABEL: test_vbsl_p16:
; CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vbsl3.i = tail call <4 x i16> @llvm.arm.neon.vbsl.v4i16(<4 x i16> %v1, <4 x i16> %v2, <4 x i16> %v3)
  ret <4 x i16> %vbsl3.i
}

define <16 x i8> @test_vbslq_s8(<16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3) {
; CHECK-LABEL: test_vbslq_s8:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl.i = tail call <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3)
  ret <16 x i8> %vbsl.i
}

define <8 x i16> @test_vbslq_s16(<8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3) {
; CHECK-LABEL: test_vbslq_s16:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl3.i = tail call <8 x i16> @llvm.arm.neon.vbsl.v8i16(<8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3)
  ret <8 x i16> %vbsl3.i
}

define <4 x i32> @test_vbslq_s32(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3) {
; CHECK-LABEL: test_vbslq_s32:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl3.i = tail call <4 x i32> @llvm.arm.neon.vbsl.v4i32(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3)
  ret <4 x i32> %vbsl3.i
}

define <2 x i64> @test_vbslq_s64(<2 x i64> %v1, <2 x i64> %v2, <2 x i64> %v3) {
; CHECK-LABEL: test_vbslq_s64:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl3.i = tail call <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64> %v1, <2 x i64> %v2, <2 x i64> %v3)
  ret <2 x i64> %vbsl3.i
}

define <16 x i8> @test_vbslq_u8(<16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3) {
; CHECK-LABEL: test_vbslq_u8:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl.i = tail call <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3)
  ret <16 x i8> %vbsl.i
}

define <8 x i16> @test_vbslq_u16(<8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3) {
; CHECK-LABEL: test_vbslq_u16:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl3.i = tail call <8 x i16> @llvm.arm.neon.vbsl.v8i16(<8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3)
  ret <8 x i16> %vbsl3.i
}

define <4 x i32> @test_vbslq_u32(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3) {
; CHECK-LABEL: test_vbslq_u32:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl3.i = tail call <4 x i32> @llvm.arm.neon.vbsl.v4i32(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3)
  ret <4 x i32> %vbsl3.i
}

define <2 x i64> @test_vbslq_u64(<2 x i64> %v1, <2 x i64> %v2, <2 x i64> %v3) {
; CHECK-LABEL: test_vbslq_u64:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl3.i = tail call <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64> %v1, <2 x i64> %v2, <2 x i64> %v3)
  ret <2 x i64> %vbsl3.i
}

define <4 x float> @test_vbslq_f32(<4 x i32> %v1, <4 x float> %v2, <4 x float> %v3) {
; CHECK-LABEL: test_vbslq_f32:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl.i = bitcast <4 x i32> %v1 to <4 x float>
  %vbsl3.i = tail call <4 x float> @llvm.arm.neon.vbsl.v4f32(<4 x float> %vbsl.i, <4 x float> %v2, <4 x float> %v3)
  ret <4 x float> %vbsl3.i
}

define <16 x i8> @test_vbslq_p8(<16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3) {
; CHECK-LABEL: test_vbslq_p8:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl.i = tail call <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3)
  ret <16 x i8> %vbsl.i
}

define <8 x i16> @test_vbslq_p16(<8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3) {
; CHECK-LABEL: test_vbslq_p16:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl3.i = tail call <8 x i16> @llvm.arm.neon.vbsl.v8i16(<8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3)
  ret <8 x i16> %vbsl3.i
}

define <2 x double> @test_vbslq_f64(<2 x i64> %v1, <2 x double> %v2, <2 x double> %v3) {
; CHECK-LABEL: test_vbslq_f64:
; CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %vbsl.i = bitcast <2 x i64> %v1 to <2 x double>
  %vbsl3.i = tail call <2 x double> @llvm.arm.neon.vbsl.v2f64(<2 x double> %vbsl.i, <2 x double> %v2, <2 x double> %v3)
  ret <2 x double> %vbsl3.i
}

