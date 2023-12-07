; RUN: opt < %s -passes='function(loop-simplify),loop-extract'  -S | FileCheck %s

@choum.addr = internal unnamed_addr constant [3 x ptr] [ptr blockaddress(@choum, %bb10), ptr blockaddress(@choum, %bb14), ptr blockaddress(@choum, %bb18)]

; CHECK: define
; no outlined function
; CHECK-NOT: define

define void @choum(i32 %arg, ptr nocapture %arg1, i32 %arg2) {
bb:
  %tmp = icmp sgt i32 %arg, 0
  br i1 %tmp, label %bb3, label %bb24

bb3:                                              ; preds = %bb
  %tmp4 = sext i32 %arg2 to i64
  %tmp5 = getelementptr inbounds [3 x ptr], ptr @choum.addr, i64 0, i64 %tmp4
  %tmp6 = load ptr, ptr %tmp5
  %tmp7 = zext i32 %arg to i64
  br label %bb8

bb8:                                              ; preds = %bb18, %bb3
  %tmp9 = phi i64 [ 0, %bb3 ], [ %tmp22, %bb18 ]
  indirectbr ptr %tmp6, [label %bb10, label %bb14, label %bb18]

bb10:                                             ; preds = %bb8
  %tmp11 = getelementptr inbounds i32, ptr %arg1, i64 %tmp9
  %tmp12 = load i32, ptr %tmp11
  %tmp13 = add nsw i32 %tmp12, 1
  store i32 %tmp13, ptr %tmp11
  br label %bb14

bb14:                                             ; preds = %bb10, %bb8
  %tmp15 = getelementptr inbounds i32, ptr %arg1, i64 %tmp9
  %tmp16 = load i32, ptr %tmp15
  %tmp17 = shl nsw i32 %tmp16, 1
  store i32 %tmp17, ptr %tmp15
  br label %bb18

bb18:                                             ; preds = %bb14, %bb8
  %tmp19 = getelementptr inbounds i32, ptr %arg1, i64 %tmp9
  %tmp20 = load i32, ptr %tmp19
  %tmp21 = add nsw i32 %tmp20, -3
  store i32 %tmp21, ptr %tmp19
  %tmp22 = add nuw nsw i64 %tmp9, 1
  %tmp23 = icmp eq i64 %tmp22, %tmp7
  br i1 %tmp23, label %bb24, label %bb8

bb24:                                             ; preds = %bb18, %bb
  ret void
}
