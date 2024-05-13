; RUN: opt -S -passes=loop-fusion < %s | FileCheck %s

@A = common global [1024 x i32] zeroinitializer, align 16
@B = common global [1024 x i32] zeroinitializer, align 16
@C = common global [1024 x i32] zeroinitializer, align 16
@D = common global [1024 x i32] zeroinitializer, align 16

; CHECK: void @dep_free
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]+]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br label %[[LOOP2BODY:bb[0-9]+]]
; CHECK: [[LOOP2BODY]]
; CHECK: br label %[[LOOP3BODY:bb[0-9]+]]
; CHECK: [[LOOP3BODY]]
; CHECK: br label %[[LOOP4BODY:bb[0-9]+]]
; CHECK: [[LOOP4BODY]]
; CHECK: br label %[[LOOP1LATCH:bb[0-9]+]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1HEADER]], label %[[LOOPEXIT:bb[0-9]+]]
; CHECK: ret void
define void @dep_free() {
bb:
  br label %bb15

bb25.preheader:                                   ; preds = %bb22
  br label %bb27

bb15:                                             ; preds = %bb, %bb22
  %.08 = phi i32 [ 0, %bb ], [ %tmp23, %bb22 ]
  %indvars.iv107 = phi i64 [ 0, %bb ], [ %indvars.iv.next11, %bb22 ]
  %tmp = add nsw i32 %.08, -3
  %tmp16 = add nuw nsw i64 %indvars.iv107, 3
  %tmp17 = trunc i64 %tmp16 to i32
  %tmp18 = mul nsw i32 %tmp, %tmp17
  %tmp19 = trunc i64 %indvars.iv107 to i32
  %tmp20 = srem i32 %tmp18, %tmp19
  %tmp21 = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv107
  store i32 %tmp20, ptr %tmp21, align 4
  br label %bb22

bb22:                                             ; preds = %bb15
  %indvars.iv.next11 = add nuw nsw i64 %indvars.iv107, 1
  %tmp23 = add nuw nsw i32 %.08, 1
  %exitcond12 = icmp ne i64 %indvars.iv.next11, 100
  br i1 %exitcond12, label %bb15, label %bb25.preheader

bb38.preheader:                                   ; preds = %bb35
  br label %bb40

bb27:                                             ; preds = %bb25.preheader, %bb35
  %.016 = phi i32 [ 0, %bb25.preheader ], [ %tmp36, %bb35 ]
  %indvars.iv75 = phi i64 [ 0, %bb25.preheader ], [ %indvars.iv.next8, %bb35 ]
  %tmp28 = add nsw i32 %.016, -3
  %tmp29 = add nuw nsw i64 %indvars.iv75, 3
  %tmp30 = trunc i64 %tmp29 to i32
  %tmp31 = mul nsw i32 %tmp28, %tmp30
  %tmp32 = trunc i64 %indvars.iv75 to i32
  %tmp33 = srem i32 %tmp31, %tmp32
  %tmp34 = getelementptr inbounds [1024 x i32], ptr @B, i64 0, i64 %indvars.iv75
  store i32 %tmp33, ptr %tmp34, align 4
  br label %bb35

bb35:                                             ; preds = %bb27
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv75, 1
  %tmp36 = add nuw nsw i32 %.016, 1
  %exitcond9 = icmp ne i64 %indvars.iv.next8, 100
  br i1 %exitcond9, label %bb27, label %bb38.preheader

bb51.preheader:                                   ; preds = %bb48
  br label %bb53

bb40:                                             ; preds = %bb38.preheader, %bb48
  %.024 = phi i32 [ 0, %bb38.preheader ], [ %tmp49, %bb48 ]
  %indvars.iv43 = phi i64 [ 0, %bb38.preheader ], [ %indvars.iv.next5, %bb48 ]
  %tmp41 = add nsw i32 %.024, -3
  %tmp42 = add nuw nsw i64 %indvars.iv43, 3
  %tmp43 = trunc i64 %tmp42 to i32
  %tmp44 = mul nsw i32 %tmp41, %tmp43
  %tmp45 = trunc i64 %indvars.iv43 to i32
  %tmp46 = srem i32 %tmp44, %tmp45
  %tmp47 = getelementptr inbounds [1024 x i32], ptr @C, i64 0, i64 %indvars.iv43
  store i32 %tmp46, ptr %tmp47, align 4
  br label %bb48

bb48:                                             ; preds = %bb40
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv43, 1
  %tmp49 = add nuw nsw i32 %.024, 1
  %exitcond6 = icmp ne i64 %indvars.iv.next5, 100
  br i1 %exitcond6, label %bb40, label %bb51.preheader

bb52:                                             ; preds = %bb61
  br label %bb63

bb53:                                             ; preds = %bb51.preheader, %bb61
  %.032 = phi i32 [ 0, %bb51.preheader ], [ %tmp62, %bb61 ]
  %indvars.iv1 = phi i64 [ 0, %bb51.preheader ], [ %indvars.iv.next, %bb61 ]
  %tmp54 = add nsw i32 %.032, -3
  %tmp55 = add nuw nsw i64 %indvars.iv1, 3
  %tmp56 = trunc i64 %tmp55 to i32
  %tmp57 = mul nsw i32 %tmp54, %tmp56
  %tmp58 = trunc i64 %indvars.iv1 to i32
  %tmp59 = srem i32 %tmp57, %tmp58
  %tmp60 = getelementptr inbounds [1024 x i32], ptr @D, i64 0, i64 %indvars.iv1
  store i32 %tmp59, ptr %tmp60, align 4
  br label %bb61

bb61:                                             ; preds = %bb53
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %tmp62 = add nuw nsw i32 %.032, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %bb53, label %bb52

bb63:                                             ; preds = %bb52
  ret void
}
