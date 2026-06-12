; Reduced from https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=65323 by @RKSimon
;
; RUN: opt -S -passes=loop-reduce %s | FileCheck %s
;
; Make sure we don't trigger an assertion.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@G = external global i32

define void @foo() {
; CHECK-LABEL: foo
bb8:
  br label %bb30

bb30:                                             ; preds = %bb30, %bb8
  %l0 = phi i64 [ -2222, %bb8 ], [ %r23, %bb30 ]
  %A22 = alloca i16, align 2
  %r23 = add nuw i64 1, %l0
  %G7 = getelementptr i16, ptr %A22, i64 %r23
  %B15 = urem i64 %r23, %r23
  %G6 = getelementptr i16, ptr %G7, i64 %B15
  %B1 = urem i64 %r23, %r23
  %B8 = sub i64 -1, %r23
  %B18 = sub i64 %B8, %B1
  %G5 = getelementptr i16, ptr %G6, i64 %B18
  store ptr %G5, ptr undef, align 8
  br label %bb30
}
