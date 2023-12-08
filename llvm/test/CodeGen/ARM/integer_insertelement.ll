; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

; This test checks that when inserting one (integer) element into a vector,
; the vector is not spuriously copied. "vorr dX, dY, dY" is the way of moving
; one DPR to another that we check for.

; CHECK: @f
; CHECK-NOT: vorr d
; CHECK: vmov.32 d
; CHECK-NOT: vorr d
; CHECK: mov pc, lr
define <4 x i32> @f(<4 x i32> %in) {
  %1 = insertelement <4 x i32> %in, i32 255, i32 3
  ret <4 x i32> %1
}

; CHECK: @g
; CHECK-NOT: vorr d
; CHECK: vmov.16 d
; CHECK-NOT: vorr d
; CHECK: mov pc, lr
define <8 x i16> @g(<8 x i16> %in) {
  %1 = insertelement <8 x i16> %in, i16 255, i32 7
  ret <8 x i16> %1
}

; CHECK: @h
; CHECK-NOT: vorr d
; CHECK: vmov.8 d
; CHECK-NOT: vorr d
; CHECK: mov pc, lr
define <16 x i8> @h(<16 x i8> %in) {
  %1 = insertelement <16 x i8> %in, i8 255, i32 15
  ret <16 x i8> %1
}
