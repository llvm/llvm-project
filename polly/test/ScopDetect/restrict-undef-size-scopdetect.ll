; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s
; CHECK-NOT: Valid Region for Scop:

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.bar = type { i32, [4 x i32] }

define void @f(ptr %arg) {
bb:
  %tmp = alloca [4 x i32], align 16
  br label %bb1

bb1:                                              ; preds = %bb8, %bb
  %tmp2 = phi i64 [ 0, %bb ], [ %tmp9, %bb8 ]
  br i1 false, label %bb3, label %bb6

bb3:                                              ; preds = %bb1
  %tmp5 = load i32, ptr %tmp
  br label %bb8

bb6:                                              ; preds = %bb1
  %tmp7 = getelementptr inbounds %struct.bar, ptr %arg, i64 0, i32 1, i64 undef
  store i32 42, ptr %tmp7
  br label %bb8

bb8:                                              ; preds = %bb6, %bb3
  %tmp9 = add nuw nsw i64 %tmp2, 1
  br i1 false, label %bb1, label %bb10

bb10:                                             ; preds = %bb8
  ret void
}
