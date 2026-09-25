; RUN: opt -passes='function(scalarizer),module(dxil-op-lower)' -S -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Lower the dx.imul/dx.umul intrinsics to the DXIL IMul(41)/UMul(42) ops, which
; return the high and low halves of the full-width product as two i32s.

define i32 @umul_scalar(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @umul_scalar(
; CHECK-SAME: i32 [[A:%.*]], i32 [[B:%.*]]) {
; CHECK-NEXT:    [[M:%.*]] = call [[DX_TYPES_TWOI32:%.*]] @dx.op.binaryWithTwoOuts.i32(i32 42, i32 [[A]], i32 [[B]])
; CHECK-NEXT:    [[HI:%.*]] = extractvalue [[DX_TYPES_TWOI32]] [[M]], 0
; CHECK-NEXT:    [[LO:%.*]] = extractvalue [[DX_TYPES_TWOI32]] [[M]], 1
; CHECK-NEXT:    [[R:%.*]] = add i32 [[HI]], [[LO]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %m = call { i32, i32 } @llvm.dx.umul.i32(i32 %a, i32 %b)
  %hi = extractvalue { i32, i32 } %m, 0
  %lo = extractvalue { i32, i32 } %m, 1
  %r = add i32 %hi, %lo
  ret i32 %r
}

define i32 @imul_scalar(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @imul_scalar(
; CHECK-SAME: i32 [[A:%.*]], i32 [[B:%.*]]) {
; CHECK-NEXT:    [[M:%.*]] = call [[DX_TYPES_TWOI32:%.*]] @dx.op.binaryWithTwoOuts.i32(i32 41, i32 [[A]], i32 [[B]])
; CHECK-NEXT:    [[HI:%.*]] = extractvalue [[DX_TYPES_TWOI32]] [[M]], 0
; CHECK-NEXT:    [[LO:%.*]] = extractvalue [[DX_TYPES_TWOI32]] [[M]], 1
; CHECK-NEXT:    [[R:%.*]] = add i32 [[HI]], [[LO]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %m = call { i32, i32 } @llvm.dx.imul.i32(i32 %a, i32 %b)
  %hi = extractvalue { i32, i32 } %m, 0
  %lo = extractvalue { i32, i32 } %m, 1
  %r = add i32 %hi, %lo
  ret i32 %r
}

; Vector calls scalarize into per-lane ops before lowering.
define <2 x i32> @umul_vector(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: define <2 x i32> @umul_vector(
; CHECK-SAME: <2 x i32> [[A:%.*]], <2 x i32> [[B:%.*]]) {
; CHECK-NEXT:    [[A0:%.*]] = extractelement <2 x i32> [[A]], i64 0
; CHECK-NEXT:    [[B0:%.*]] = extractelement <2 x i32> [[B]], i64 0
; CHECK-NEXT:    [[M0:%.*]] = call [[DX_TYPES_TWOI32:%.*]] @dx.op.binaryWithTwoOuts.i32(i32 42, i32 [[A0]], i32 [[B0]])
; CHECK-NEXT:    [[A1:%.*]] = extractelement <2 x i32> [[A]], i64 1
; CHECK-NEXT:    [[B1:%.*]] = extractelement <2 x i32> [[B]], i64 1
; CHECK-NEXT:    [[M1:%.*]] = call [[DX_TYPES_TWOI32]] @dx.op.binaryWithTwoOuts.i32(i32 42, i32 [[A1]], i32 [[B1]])
; CHECK-NEXT:    [[LO0:%.*]] = extractvalue [[DX_TYPES_TWOI32]] [[M0]], 1
; CHECK-NEXT:    [[LO1:%.*]] = extractvalue [[DX_TYPES_TWOI32]] [[M1]], 1
; CHECK-NEXT:    [[R0:%.*]] = insertelement <2 x i32> poison, i32 [[LO0]], i64 0
; CHECK-NEXT:    [[R:%.*]] = insertelement <2 x i32> [[R0]], i32 [[LO1]], i64 1
; CHECK-NEXT:    ret <2 x i32> [[R]]
;
  %m = call { <2 x i32>, <2 x i32> } @llvm.dx.umul.v2i32(<2 x i32> %a, <2 x i32> %b)
  %lo = extractvalue { <2 x i32>, <2 x i32> } %m, 1
  ret <2 x i32> %lo
}
