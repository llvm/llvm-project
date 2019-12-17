; RUN: opt -S -loop-fusion < %s | FileCheck %s

@B = common global [1024 x i32] zeroinitializer, align 16

; CHECK: void @dep_free_parametric
; CHECK-next: entry:
; CHECK: br i1 %{{.*}}, label %[[LOOP1PREHEADER:bb[0-9]*]], label %[[LOOP1SUCC:bb[0-9]+]]
; CHECK: [[LOOP1PREHEADER]]
; CHECK-NEXT: br label %[[LOOP1BODY:bb[0-9]*]]
; CHECK: [[LOOP1BODY]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1BODY]], label %[[LOOP2EXIT:bb[0-9]+]]
; CHECK: [[LOOP2EXIT]]
; CHECK: br label %[[LOOP1SUCC]]
; CHECK: [[LOOP1SUCC]]
; CHECK: ret void
define void @dep_free_parametric(i32* noalias %A, i64 %N) {
entry:
  %cmp4 = icmp slt i64 0, %N
  br i1 %cmp4, label %bb3, label %bb14

bb3:                               ; preds = %entry
  br label %bb5

bb5:                                         ; preds = %bb3, %bb5
  %i.05 = phi i64 [ %inc, %bb5 ], [ 0, %bb3 ]
  %sub = sub nsw i64 %i.05, 3
  %add = add nsw i64 %i.05, 3
  %mul = mul nsw i64 %sub, %add
  %rem = srem i64 %mul, %i.05
  %conv = trunc i64 %rem to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.05
  store i32 %conv, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.05, 1
  %cmp = icmp slt i64 %inc, %N
  br i1 %cmp, label %bb5, label %bb10

bb10:                                 ; preds = %bb5
  br label %bb14

bb14:                                          ; preds = %bb10, %entry
  %cmp31 = icmp slt i64 0, %N
  br i1 %cmp31, label %bb8, label %bb12

bb8:                              ; preds = %bb14
  br label %bb9

bb9:                                        ; preds = %bb8, %bb9
  %i1.02 = phi i64 [ %inc14, %bb9 ], [ 0, %bb8 ]
  %sub7 = sub nsw i64 %i1.02, 3
  %add8 = add nsw i64 %i1.02, 3
  %mul9 = mul nsw i64 %sub7, %add8
  %rem10 = srem i64 %mul9, %i1.02
  %conv11 = trunc i64 %rem10 to i32
  %arrayidx12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %i1.02
  store i32 %conv11, i32* %arrayidx12, align 4
  %inc14 = add nsw i64 %i1.02, 1
  %cmp3 = icmp slt i64 %inc14, %N
  br i1 %cmp3, label %bb9, label %bb15

bb15:                               ; preds = %bb9
  br label %bb12

bb12:                                        ; preds = %bb15, %bb14
  ret void
}
