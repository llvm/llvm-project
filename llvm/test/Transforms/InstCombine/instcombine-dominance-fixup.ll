; NOTE: This test ensures InstCombine preserves dominance even when it
; reorders shifts through SimplifyDemandedBits/log2 folding.
;
; RUN: opt -passes=instcombine,instnamer < %s | FileCheck %s

define i64 @f(i64 %arg0, i64 %arg1) {
; CHECK-LABEL: define i64 @f(
; CHECK-SAME: i64 [[ARG0:%[^,]+]], i64 [[ARG1:%[^)]+]])
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SHL:%[^ ]+]] = shl nuw i64 1, [[ARG1]]
; CHECK-NEXT:    [[LSHR:%[^ ]+]] = lshr exact i64 [[SHL]], 1
; CHECK-NEXT:    [[SHL2:%[^ ]+]] = shl nuw i64 [[LSHR]], [[ARG1]]
; CHECK-NEXT:    [[SREM:%[^ ]+]] = srem i64 [[ARG0]], [[SHL2]]
; CHECK-NEXT:    ret i64 [[SREM]]
entry:
  %shl = shl nuw i64 1, %arg1
  %lshr = lshr exact i64 %shl, 1
  %shl2 = shl nuw i64 %lshr, %arg1
  %srem = srem i64 %arg0, %shl2
  ret i64 %srem
}
