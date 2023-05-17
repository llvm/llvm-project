; RUN: opt %loadPolly -polly-process-unprofitable -polly-scops -disable-output < %s
;
; This test contains a infinite loop (bb13) and crashed the domain generation
; at some point. Just verify it does not anymore.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @hoge() #0 {
bb:
  %tmp5 = alloca [11 x [101 x i32]], align 16
  br i1 false, label %bb24, label %bb6

bb6:                                              ; preds = %bb
  br label %bb8

bb7:                                              ; preds = %bb23
  unreachable

bb8:                                              ; preds = %bb23, %bb6
  br label %bb10

bb10:                                             ; preds = %bb8
  %tmp = load i32, ptr %tmp5, align 4
  br i1 false, label %bb23, label %bb11

bb11:                                             ; preds = %bb10
  %tmp12 = load i32, ptr %tmp5, align 4
  br label %bb13

bb13:                                             ; preds = %bb13, %bb11
  %tmp14 = phi i32 [ %tmp12, %bb11 ], [ %tmp19, %bb13 ]
  %tmp15 = add nsw i32 %tmp14, 1
  %tmp16 = sext i32 %tmp15 to i64
  %tmp17 = getelementptr inbounds [11 x [101 x i32]], ptr %tmp5, i64 0, i64 0, i64 %tmp16
  %tmp18 = load i32, ptr %tmp17, align 4
  %tmp19 = add nsw i32 %tmp14, -1
  %tmp20 = load i32, ptr %tmp5, align 4
  %tmp21 = sext i32 %tmp20 to i64
  %tmp22 = icmp slt i64 0, %tmp21
  br label %bb13

bb23:                                             ; preds = %bb10
  br i1 undef, label %bb8, label %bb7

bb24:                                             ; preds = %bb
  ret void
}

define void @hoge2() #0 {
bb:
  %tmp5 = alloca [11 x [101 x i32]], align 16
  br i1 false, label %bb24, label %bb6

bb6:                                              ; preds = %bb
  br label %bb8

bb7:                                              ; preds = %bb23
  unreachable

bb8:                                              ; preds = %bb23, %bb6
  br label %bb10

bb10:                                             ; preds = %bb8
  %tmp = load i32, ptr %tmp5, align 4
  br i1 false, label %bb23, label %bb11

bb11:                                             ; preds = %bb10
  %tmp12 = load i32, ptr %tmp5, align 4
  br label %bb13

bb13:                                             ; preds = %bb13, %bb11
  %tmp14 = phi i32 [ %tmp12, %bb11 ], [ %tmp19, %bb13.split ]
  %tmp15 = add nsw i32 %tmp14, 1
  %tmp16 = sext i32 %tmp15 to i64
  %tmp17 = getelementptr inbounds [11 x [101 x i32]], ptr %tmp5, i64 0, i64 0, i64 %tmp16
  %tmp18 = load i32, ptr %tmp17, align 4
  br label %bb13.split

bb13.split:
  %tmp19 = add nsw i32 %tmp14, -1
  %tmp20 = load i32, ptr %tmp5, align 4
  %tmp21 = sext i32 %tmp20 to i64
  %tmp22 = icmp slt i64 0, %tmp21
  br label %bb13

bb23:                                             ; preds = %bb10
  br i1 undef, label %bb8, label %bb7

bb24:                                             ; preds = %bb
  ret void
}
