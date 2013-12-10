; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_uabd_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_uabd_v8i8:
  %abd = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: uabd v0.8b, v0.8b, v1.8b
  ret <8 x i8> %abd
}

define <8 x i8> @test_uaba_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_uaba_v8i8:
  %abd = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
  %aba = add <8 x i8> %lhs, %abd
; CHECK: uaba v0.8b, v0.8b, v1.8b
  ret <8 x i8> %aba
}

define <8 x i8> @test_sabd_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_sabd_v8i8:
  %abd = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: sabd v0.8b, v0.8b, v1.8b
  ret <8 x i8> %abd
}

define <8 x i8> @test_saba_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_saba_v8i8:
  %abd = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
  %aba = add <8 x i8> %lhs, %abd
; CHECK: saba v0.8b, v0.8b, v1.8b
  ret <8 x i8> %aba
}

declare <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_uabd_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_uabd_v16i8:
  %abd = call <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: uabd v0.16b, v0.16b, v1.16b
  ret <16 x i8> %abd
}

define <16 x i8> @test_uaba_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_uaba_v16i8:
  %abd = call <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
  %aba = add <16 x i8> %lhs, %abd
; CHECK: uaba v0.16b, v0.16b, v1.16b
  ret <16 x i8> %aba
}

define <16 x i8> @test_sabd_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_sabd_v16i8:
  %abd = call <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: sabd v0.16b, v0.16b, v1.16b
  ret <16 x i8> %abd
}

define <16 x i8> @test_saba_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_saba_v16i8:
  %abd = call <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
  %aba = add <16 x i8> %lhs, %abd
; CHECK: saba v0.16b, v0.16b, v1.16b
  ret <16 x i8> %aba
}

declare <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_uabd_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_uabd_v4i16:
  %abd = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: uabd v0.4h, v0.4h, v1.4h
  ret <4 x i16> %abd
}

define <4 x i16> @test_uaba_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_uaba_v4i16:
  %abd = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
  %aba = add <4 x i16> %lhs, %abd
; CHECK: uaba v0.4h, v0.4h, v1.4h
  ret <4 x i16> %aba
}

define <4 x i16> @test_sabd_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_sabd_v4i16:
  %abd = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: sabd v0.4h, v0.4h, v1.4h
  ret <4 x i16> %abd
}

define <4 x i16> @test_saba_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_saba_v4i16:
  %abd = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
  %aba = add <4 x i16> %lhs, %abd
; CHECK: saba v0.4h, v0.4h, v1.4h
  ret <4 x i16> %aba
}

declare <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_uabd_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_uabd_v8i16:
  %abd = call <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: uabd v0.8h, v0.8h, v1.8h
  ret <8 x i16> %abd
}

define <8 x i16> @test_uaba_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_uaba_v8i16:
  %abd = call <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
  %aba = add <8 x i16> %lhs, %abd
; CHECK: uaba v0.8h, v0.8h, v1.8h
  ret <8 x i16> %aba
}

define <8 x i16> @test_sabd_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_sabd_v8i16:
  %abd = call <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: sabd v0.8h, v0.8h, v1.8h
  ret <8 x i16> %abd
}

define <8 x i16> @test_saba_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_saba_v8i16:
  %abd = call <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
  %aba = add <8 x i16> %lhs, %abd
; CHECK: saba v0.8h, v0.8h, v1.8h
  ret <8 x i16> %aba
}

declare <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_uabd_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_uabd_v2i32:
  %abd = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: uabd v0.2s, v0.2s, v1.2s
  ret <2 x i32> %abd
}

define <2 x i32> @test_uaba_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_uaba_v2i32:
  %abd = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
  %aba = add <2 x i32> %lhs, %abd
; CHECK: uaba v0.2s, v0.2s, v1.2s
  ret <2 x i32> %aba
}

define <2 x i32> @test_sabd_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_sabd_v2i32:
  %abd = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: sabd v0.2s, v0.2s, v1.2s
  ret <2 x i32> %abd
}

define <2 x i32> @test_sabd_v2i32_const() {
; CHECK: test_sabd_v2i32_const:
; CHECK: movi     d1, #0xffffffff0000
; CHECK-NEXT: sabd v0.2s, v0.2s, v1.2s
  %1 = tail call <2 x i32> @llvm.arm.neon.vabds.v2i32(
    <2 x i32> <i32 -2147483648, i32 2147450880>,
    <2 x i32> <i32 -65536, i32 65535>)
  ret <2 x i32> %1
}

define <2 x i32> @test_saba_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_saba_v2i32:
  %abd = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
  %aba = add <2 x i32> %lhs, %abd
; CHECK: saba v0.2s, v0.2s, v1.2s
  ret <2 x i32> %aba
}

declare <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_uabd_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_uabd_v4i32:
  %abd = call <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: uabd v0.4s, v0.4s, v1.4s
  ret <4 x i32> %abd
}

define <4 x i32> @test_uaba_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_uaba_v4i32:
  %abd = call <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
  %aba = add <4 x i32> %lhs, %abd
; CHECK: uaba v0.4s, v0.4s, v1.4s
  ret <4 x i32> %aba
}

define <4 x i32> @test_sabd_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_sabd_v4i32:
  %abd = call <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: sabd v0.4s, v0.4s, v1.4s
  ret <4 x i32> %abd
}

define <4 x i32> @test_saba_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_saba_v4i32:
  %abd = call <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
  %aba = add <4 x i32> %lhs, %abd
; CHECK: saba v0.4s, v0.4s, v1.4s
  ret <4 x i32> %aba
}

declare <2 x float> @llvm.arm.neon.vabds.v2f32(<2 x float>, <2 x float>)

define <2 x float> @test_fabd_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fabd_v2f32:
  %abd = call <2 x float> @llvm.arm.neon.vabds.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fabd v0.2s, v0.2s, v1.2s
  ret <2 x float> %abd
}

declare <4 x float> @llvm.arm.neon.vabds.v4f32(<4 x float>, <4 x float>)

define <4 x float> @test_fabd_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fabd_v4f32:
  %abd = call <4 x float> @llvm.arm.neon.vabds.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fabd v0.4s, v0.4s, v1.4s
  ret <4 x float> %abd
}

declare <2 x double> @llvm.arm.neon.vabds.v2f64(<2 x double>, <2 x double>)

define <2 x double> @test_fabd_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fabd_v2f64:
  %abd = call <2 x double> @llvm.arm.neon.vabds.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fabd v0.2d, v0.2d, v1.2d
  ret <2 x double> %abd
}
