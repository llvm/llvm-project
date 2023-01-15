; RUN: opt -passes=loop-versioning -S < %s | FileCheck %s

; Make sure all PHIs are properly updated in the exit block.  Based on
; PR28037.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = external global [2 x [3 x [5 x i16]]]

; CHECK-LABEL: @phi_with_undef
define void @phi_with_undef() {
bb6.lr.ph:                                        ; preds = %bb5.preheader
  br label %bb6

bb6:                                              ; preds = %bb6.lr.ph, %bb6
  %_tmp1423 = phi i64 [ undef, %bb6.lr.ph ], [ %_tmp142, %bb6 ]
  %_tmp123 = getelementptr [2 x [3 x [5 x i16]]], ptr @x, i16 0, i64 undef
  %_tmp126 = getelementptr [3 x [5 x i16]], ptr %_tmp123, i16 0, i64 %_tmp1423
  %_tmp129 = getelementptr [5 x i16], ptr %_tmp126, i16 0, i64 undef
  %_tmp130 = load i16, ptr %_tmp129
  store i16 undef, ptr getelementptr ([2 x [3 x [5 x i16]]], ptr @x, i64 0, i64 undef, i64 undef, i64 undef)
  %_tmp142 = add i64 %_tmp1423, 1
  br i1 false, label %bb6, label %loop.exit

loop.exit:                                ; preds = %bb6
  %_tmp142.lcssa = phi i64 [ %_tmp142, %bb6 ]
  %split = phi i16 [ undef, %bb6 ]
; CHECK: %split.ph = phi i16 [ undef, %bb6.lver.orig ]
; CHECK: %split.ph3 = phi i16 [ undef, %bb6 ]
; CHECK: %split = phi i16 [ %split.ph, %loop.exit.loopexit ], [ %split.ph3, %loop.exit.loopexit1 ]
  br label %bb9

bb9:                                              ; preds = %bb9.loopexit, %bb1
  ret void
}

; CHECK-LABEL: @phi_with_non_loop_defined_value
define void @phi_with_non_loop_defined_value() {
bb6.lr.ph:                                        ; preds = %bb5.preheader
  %t = add i16 1, 1
  br label %bb6

bb6:                                              ; preds = %bb6.lr.ph, %bb6
  %_tmp1423 = phi i64 [ undef, %bb6.lr.ph ], [ %_tmp142, %bb6 ]
  %_tmp123 = getelementptr [2 x [3 x [5 x i16]]], ptr @x, i16 0, i64 undef
  %_tmp126 = getelementptr [3 x [5 x i16]], ptr %_tmp123, i16 0, i64 %_tmp1423
  %_tmp129 = getelementptr [5 x i16], ptr %_tmp126, i16 0, i64 undef
  %_tmp130 = load i16, ptr %_tmp129
  store i16 undef, ptr getelementptr ([2 x [3 x [5 x i16]]], ptr @x, i64 0, i64 undef, i64 undef, i64 undef)
  %_tmp142 = add i64 %_tmp1423, 1
  br i1 false, label %bb6, label %loop.exit

loop.exit:                                ; preds = %bb6
  %_tmp142.lcssa = phi i64 [ %_tmp142, %bb6 ]
  %split = phi i16 [ %t, %bb6 ]
; CHECK: %split.ph = phi i16 [ %t, %bb6.lver.orig ]
; CHECK: %split.ph3 = phi i16 [ %t, %bb6 ]
; CHECK: %split = phi i16 [ %split.ph, %loop.exit.loopexit ], [ %split.ph3, %loop.exit.loopexit1 ]
  br label %bb9

bb9:                                              ; preds = %bb9.loopexit, %bb1
  ret void
}
