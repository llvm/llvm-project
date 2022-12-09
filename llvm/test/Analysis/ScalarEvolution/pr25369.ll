; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hoge1() {
; CHECK-LABEL: Classifying expressions for: @hoge1
bb:
  br i1 undef, label %bb4, label %bb2

bb2:                                              ; preds = %bb2, %bb
  br i1 false, label %bb4, label %bb2

bb3:                                              ; preds = %bb4
  %tmp = add i32 %tmp10, -1
  br label %bb13

bb4:                                              ; preds = %bb4, %bb2, %bb
  %tmp5 = phi i64 [ %tmp11, %bb4 ], [ 1, %bb2 ], [ 1, %bb ]
  %tmp6 = phi i32 [ %tmp10, %bb4 ], [ 0, %bb2 ], [ 0, %bb ]
  %tmp7 = load i32, i32* undef, align 4
  %tmp8 = add i32 %tmp7, %tmp6
  %tmp9 = add i32 undef, %tmp8
  %tmp10 = add i32 undef, %tmp9
  %tmp11 = add nsw i64 %tmp5, 3
  %tmp12 = icmp eq i64 %tmp11, 64
  br i1 %tmp12, label %bb3, label %bb4

; CHECK: Loop %bb4: backedge-taken count is 20
; CHECK: Loop %bb4: constant max backedge-taken count is 20

bb13:                                             ; preds = %bb13, %bb3
  %tmp14 = phi i64 [ 0, %bb3 ], [ %tmp15, %bb13 ]
  %tmp15 = add nuw nsw i64 %tmp14, 1
  %tmp16 = trunc i64 %tmp15 to i32
  %tmp17 = icmp eq i32 %tmp16, %tmp
  br i1 %tmp17, label %bb18, label %bb13

bb18:                                             ; preds = %bb13
  ret void
}

define void @hoge2() {
; CHECK-LABEL: Classifying expressions for: @hoge2
bb:
  br i1 undef, label %bb4, label %bb2

bb2:                                              ; preds = %bb2, %bb
  br i1 false, label %bb4, label %bb2

bb3:                                              ; preds = %bb4
  %tmp = add i32 %tmp10, -1
  br label %bb13

bb4:                                              ; preds = %bb4, %bb2, %bb
  %tmp5 = phi i64 [ %tmp11, %bb4 ], [ 1, %bb2 ], [ 3, %bb ]
  %tmp6 = phi i32 [ %tmp10, %bb4 ], [ 0, %bb2 ], [ 0, %bb ]
  %tmp7 = load i32, i32* undef, align 4
  %tmp8 = add i32 %tmp7, %tmp6
  %tmp9 = add i32 undef, %tmp8
  %tmp10 = add i32 undef, %tmp9
  %tmp11 = add nsw i64 %tmp5, 3
  %tmp12 = icmp eq i64 %tmp11, 64
  br i1 %tmp12, label %bb3, label %bb4

; CHECK: Loop %bb4: Unpredictable backedge-taken count.
; CHECK: Loop %bb4: Unpredictable constant max backedge-taken count.

bb13:                                             ; preds = %bb13, %bb3
  %tmp14 = phi i64 [ 0, %bb3 ], [ %tmp15, %bb13 ]
  %tmp15 = add nuw nsw i64 %tmp14, 1
  %tmp16 = trunc i64 %tmp15 to i32
  %tmp17 = icmp eq i32 %tmp16, %tmp
  br i1 %tmp17, label %bb18, label %bb13

bb18:                                             ; preds = %bb13
  ret void
}
