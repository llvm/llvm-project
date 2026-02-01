; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-musl -global-isel -global-isel-abort=1 < %s | FileCheck %s

define <2 x i8> @test_bitcast_assertion(<4 x i32> %vqaddq_v2.i.i, ptr %BS_VAR_0) {
; CHECK-LABEL: test_bitcast_assertion:
; CHECK:       sub sp, sp, #16
; CHECK:       movi v[[ZERO_REG:[0-9]+]].2d, #0
; CHECK:       mov [[PTR_TMP:x[0-9]+]], sp

; CHECK:       .LBB0_1: // %for.cond
; CHECK:       umov [[EXTRACTED:w[0-9]+]], v[[ZERO_REG]].h[0]
; CHECK:       str q0, [sp]
; CHECK:       umull [[IDX:x[0-9]+]], [[EXTRACTED]], w9
; CHECK:       ldrh w[[VAL_REG:[0-9]+]], [[[PTR_TMP]], [[IDX]]]
; CHECK:       stp q[[ZERO_REG]], q[[ZERO_REG]], [x0, #32]
; CHECK:       stp q[[ZERO_REG]], q[[ZERO_REG]], [x0, #64]
; CHECK:       fmov d[[RES_REG:[0-9]+]], x[[VAL_REG]]
; CHECK:       stp q[[ZERO_REG]], q[[ZERO_REG]], [x0, #96]
; CHECK:       mov v[[RES_REG]].d[1], xzr
; CHECK:       stp q[[RES_REG]], q[[ZERO_REG]], [x0]
; CHECK:       b .LBB0_1

entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %0 = phi <64 x i16> [ %2, %for.cond ], [ zeroinitializer, %entry ]
  %conv = extractelement <64 x i16> %0, i64 0
  %vecext.i = extractelement <4 x i32> %vqaddq_v2.i.i, i16 %conv
  %1 = and i32 %vecext.i, 65535
  %conv1 = zext i32 %1 to i64
  %vecinit16 = insertelement <16 x i64> zeroinitializer, i64 %conv1, i64 0
  store <16 x i64> %vecinit16, ptr %BS_VAR_0, align 16
  %2 = bitcast <16 x i64> zeroinitializer to <64 x i16>
  br label %for.cond
}


