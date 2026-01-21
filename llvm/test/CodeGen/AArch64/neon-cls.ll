; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck %s

; Test vector CLS for 8-bit elements in 64-bit register
define <8 x i8> @test_cls_v8i8(<8 x i8> %x) {
; CHECK-LABEL: test_cls_v8i8:
; CHECK: cls v0.8b, v0.8b
  %result = call <8 x i8> @llvm.aarch64.neon.cls.v8i8(<8 x i8> %x)
  ret <8 x i8> %result
}

; Test vector CLS for 8-bit elements in 128-bit register
define <16 x i8> @test_cls_v16i8(<16 x i8> %x) {
; CHECK-LABEL: test_cls_v16i8:
; CHECK: cls v0.16b, v0.16b
  %result = call <16 x i8> @llvm.aarch64.neon.cls.v16i8(<16 x i8> %x)
  ret <16 x i8> %result
}

; Test vector CLS for 16-bit elements in 64-bit register
define <4 x i16> @test_cls_v4i16(<4 x i16> %x) {
; CHECK-LABEL: test_cls_v4i16:
; CHECK: cls v0.4h, v0.4h
  %result = call <4 x i16> @llvm.aarch64.neon.cls.v4i16(<4 x i16> %x)
  ret <4 x i16> %result
}

; Test vector CLS for 16-bit elements in 128-bit register
define <8 x i16> @test_cls_v8i16(<8 x i16> %x) {
; CHECK-LABEL: test_cls_v8i16:
; CHECK: cls v0.8h, v0.8h
  %result = call <8 x i16> @llvm.aarch64.neon.cls.v8i16(<8 x i16> %x)
  ret <8 x i16> %result
}

; Test vector CLS for 32-bit elements in 64-bit register
define <2 x i32> @test_cls_v2i32(<2 x i32> %x) {
; CHECK-LABEL: test_cls_v2i32:
; CHECK: cls v0.2s, v0.2s
  %result = call <2 x i32> @llvm.aarch64.neon.cls.v2i32(<2 x i32> %x)
  ret <2 x i32> %result
}

; Test vector CLS for 32-bit elements in 128-bit register
define <4 x i32> @test_cls_v4i32(<4 x i32> %x) {
; CHECK-LABEL: test_cls_v4i32:
; CHECK: cls v0.4s, v0.4s
  %result = call <4 x i32> @llvm.aarch64.neon.cls.v4i32(<4 x i32> %x)
  ret <4 x i32> %result
}

; Test multiple calls to verify no code sharing issues
define <4 x i32> @test_cls_multiple(<4 x i32> %x, <4 x i32> %y) {
; CHECK-LABEL: test_cls_multiple:
; CHECK: cls v0.4s, v0.4s
; CHECK: cls v1.4s, v1.4s
; CHECK: add v0.4s, v0.4s, v1.4s
  %cx = call <4 x i32> @llvm.aarch64.neon.cls.v4i32(<4 x i32> %x)
  %cy = call <4 x i32> @llvm.aarch64.neon.cls.v4i32(<4 x i32> %y)
  %result = add <4 x i32> %cx, %cy
  ret <4 x i32> %result
}

declare <8 x i8> @llvm.aarch64.neon.cls.v8i8(<8 x i8>)
declare <16 x i8> @llvm.aarch64.neon.cls.v16i8(<16 x i8>)
declare <4 x i16> @llvm.aarch64.neon.cls.v4i16(<4 x i16>)
declare <8 x i16> @llvm.aarch64.neon.cls.v8i16(<8 x i16>)
declare <2 x i32> @llvm.aarch64.neon.cls.v2i32(<2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.cls.v4i32(<4 x i32>)
