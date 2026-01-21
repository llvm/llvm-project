; RUN: llc -mtriple=armv7-linux-gnueabihf -mattr=+neon < %s | FileCheck %s

; Test vector CLS for 8-bit elements
define <8 x i8> @test_cls_v8i8(<8 x i8> %x) {
; CHECK-LABEL: test_cls_v8i8:
; CHECK: vcls.s8 d0, d0
  %result = call <8 x i8> @llvm.arm.neon.vcls.v8i8(<8 x i8> %x)
  ret <8 x i8> %result
}

define <16 x i8> @test_cls_v16i8(<16 x i8> %x) {
; CHECK-LABEL: test_cls_v16i8:
; CHECK: vcls.s8 q0, q0
  %result = call <16 x i8> @llvm.arm.neon.vcls.v16i8(<16 x i8> %x)
  ret <16 x i8> %result
}

; Test vector CLS for 16-bit elements
define <4 x i16> @test_cls_v4i16(<4 x i16> %x) {
; CHECK-LABEL: test_cls_v4i16:
; CHECK: vcls.s16 d0, d0
  %result = call <4 x i16> @llvm.arm.neon.vcls.v4i16(<4 x i16> %x)
  ret <4 x i16> %result
}

define <8 x i16> @test_cls_v8i16(<8 x i16> %x) {
; CHECK-LABEL: test_cls_v8i16:
; CHECK: vcls.s16 q0, q0
  %result = call <8 x i16> @llvm.arm.neon.vcls.v8i16(<8 x i16> %x)
  ret <8 x i16> %result
}

; Test vector CLS for 32-bit elements
define <2 x i32> @test_cls_v2i32(<2 x i32> %x) {
; CHECK-LABEL: test_cls_v2i32:
; CHECK: vcls.s32 d0, d0
  %result = call <2 x i32> @llvm.arm.neon.vcls.v2i32(<2 x i32> %x)
  ret <2 x i32> %result
}

define <4 x i32> @test_cls_v4i32(<4 x i32> %x) {
; CHECK-LABEL: test_cls_v4i32:
; CHECK: vcls.s32 q0, q0
  %result = call <4 x i32> @llvm.arm.neon.vcls.v4i32(<4 x i32> %x)
  ret <4 x i32> %result
}

declare <8 x i8> @llvm.arm.neon.vcls.v8i8(<8 x i8>)
declare <16 x i8> @llvm.arm.neon.vcls.v16i8(<16 x i8>)
declare <4 x i16> @llvm.arm.neon.vcls.v4i16(<4 x i16>)
declare <8 x i16> @llvm.arm.neon.vcls.v8i16(<8 x i16>)
declare <2 x i32> @llvm.arm.neon.vcls.v2i32(<2 x i32>)
declare <4 x i32> @llvm.arm.neon.vcls.v4i32(<4 x i32>)
