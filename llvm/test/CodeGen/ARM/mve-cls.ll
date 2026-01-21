; RUN: llc -mtriple=thumbv8.1m.main-none-eabi -mattr=+mve < %s | FileCheck %s

; Test vector CLS for 8-bit elements
define arm_aapcs_vfpcc <16 x i8> @test_cls_v16i8(<16 x i8> %x) {
; CHECK-LABEL: test_cls_v16i8:
; CHECK: vcls.s8 q0, q0
  %result = call <16 x i8> @llvm.arm.mve.vcls.v16i8(<16 x i8> %x)
  ret <16 x i8> %result
}

; Test vector CLS for 16-bit elements
define arm_aapcs_vfpcc <8 x i16> @test_cls_v8i16(<8 x i16> %x) {
; CHECK-LABEL: test_cls_v8i16:
; CHECK: vcls.s16 q0, q0
  %result = call <8 x i16> @llvm.arm.mve.vcls.v8i16(<8 x i16> %x)
  ret <8 x i16> %result
}

; Test vector CLS for 32-bit elements
define arm_aapcs_vfpcc <4 x i32> @test_cls_v4i32(<4 x i32> %x) {
; CHECK-LABEL: test_cls_v4i32:
; CHECK: vcls.s32 q0, q0
  %result = call <4 x i32> @llvm.arm.mve.vcls.v4i32(<4 x i32> %x)
  ret <4 x i32> %result
}

declare <16 x i8> @llvm.arm.mve.vcls.v16i8(<16 x i8>)
declare <8 x i16> @llvm.arm.mve.vcls.v8i16(<8 x i16>)
declare <4 x i32> @llvm.arm.mve.vcls.v4i32(<4 x i32>)
