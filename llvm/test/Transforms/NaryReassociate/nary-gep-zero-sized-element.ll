; RUN: opt < %s -passes=nary-reassociate -S | FileCheck %s

; This test checks that NaryReassociate does not crash when a GEP's result
; element type has a size of zero (eg. [0 x ptr] such as `%T` below).

; CHECK-LABEL: define ptr @foo
; CHECK-SAME: (ptr [[BASE:%.*]], i64 [[X:%.*]], i64 [[Y:%.*]]) {
; CHECK-NEXT:    [[Z:%.*]] = add i64 [[X]], [[Y]]
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr [[T:%.*]], ptr [[BASE]], i64 [[X]], i32 1
; CHECK-NEXT:    [[GEP2:%.*]] = getelementptr inbounds [[T]], ptr [[BASE]], i64 [[Z]], i32 1
; CHECK-NEXT:    ret ptr [[GEP2]]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%T = type { [1 x ptr], [0 x ptr] }

define ptr @foo(ptr %base, i64 %x, i64 %y) {
  %z = add i64 %x, %y
  %gep1 = getelementptr inbounds %T, ptr %base, i64 %x, i32 1
  %gep2 = getelementptr inbounds %T, ptr %base, i64 %z, i32 1
  ret ptr %gep2
}
