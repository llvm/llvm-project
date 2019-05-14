; RUN: opt -S -loop-fusion < %s | FileCheck %s

@A = common global [1024 x i32] zeroinitializer, align 16
@B = common global [1024 x i32] zeroinitializer, align 16
@C = common global [1024 x i32] zeroinitializer, align 16
@D = common global [1024 x i32] zeroinitializer, align 16

; CHECK: void @dep_free
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]+]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %exitcond12, label %[[LOOP1BODY:bb[0-9]+]], label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP1BODY]]
; CHECK: br label %[[LOOP1LATCH:bb[0-9]+]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br label %[[LOOP2PREHEADER]]
; CHECK: [[LOOP2PREHEADER]]
; CHECK: br i1 %exitcond9, label %[[LOOP2HEADER:bb[0-9]+]], label %[[LOOP3PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP2HEADER]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br label %[[LOOP3PREHEADER]]
; CHECK: [[LOOP3PREHEADER]]
; CHECK: br i1 %exitcond6, label %[[LOOP3HEADER:bb[0-9]+]], label %[[LOOP4PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP3HEADER]]
; CHECK: br label %[[LOOP3LATCH:bb[0-9]+]]
; CHECK: [[LOOP3LATCH]]
; CHECK: br label %[[LOOP4PREHEADER]]
; CHECK: [[LOOP4PREHEADER]]
; CHECK: br i1 %exitcond, label %[[LOOP4HEADER:bb[0-9]+]], label %[[LOOP4EXIT:bb[0-9]+]]
; CHECK: [[LOOP4EXIT]]
; CHECK: br label %[[FUNCEXIT:bb[0-9]+]]
; CHECK: [[LOOP4HEADER]]
; CHECK: br label %[[LOOP4LATCH:bb[0-9]+]]
; CHECK: [[LOOP4LATCH]]
; CHECK: br label %[[LOOP1HEADER]]
; CHECK: [[FUNCEXIT]]
; CHECK: ret void
define void @dep_free() {
bb:
  br label %bb13

bb13:                                             ; preds = %bb22, %bb
  %indvars.iv10 = phi i64 [ %indvars.iv.next11, %bb22 ], [ 0, %bb ]
  %.0 = phi i32 [ 0, %bb ], [ %tmp23, %bb22 ]
  %exitcond12 = icmp ne i64 %indvars.iv10, 100
  br i1 %exitcond12, label %bb15, label %bb25

bb15:                                             ; preds = %bb13
  %tmp = add nsw i32 %.0, -3
  %tmp16 = add nuw nsw i64 %indvars.iv10, 3
  %tmp17 = trunc i64 %tmp16 to i32
  %tmp18 = mul nsw i32 %tmp, %tmp17
  %tmp19 = trunc i64 %indvars.iv10 to i32
  %tmp20 = srem i32 %tmp18, %tmp19
  %tmp21 = getelementptr inbounds [1024 x i32], [1024 x i32]* @A, i64 0, i64 %indvars.iv10
  store i32 %tmp20, i32* %tmp21, align 4
  br label %bb22

bb22:                                             ; preds = %bb15
  %indvars.iv.next11 = add nuw nsw i64 %indvars.iv10, 1
  %tmp23 = add nuw nsw i32 %.0, 1
  br label %bb13

bb25:                                             ; preds = %bb35, %bb13
  %indvars.iv7 = phi i64 [ %indvars.iv.next8, %bb35 ], [ 0, %bb13 ]
  %.01 = phi i32 [ 0, %bb13 ], [ %tmp36, %bb35 ]
  %exitcond9 = icmp ne i64 %indvars.iv7, 100
  br i1 %exitcond9, label %bb27, label %bb38

bb27:                                             ; preds = %bb25
  %tmp28 = add nsw i32 %.01, -3
  %tmp29 = add nuw nsw i64 %indvars.iv7, 3
  %tmp30 = trunc i64 %tmp29 to i32
  %tmp31 = mul nsw i32 %tmp28, %tmp30
  %tmp32 = trunc i64 %indvars.iv7 to i32
  %tmp33 = srem i32 %tmp31, %tmp32
  %tmp34 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv7
  store i32 %tmp33, i32* %tmp34, align 4
  br label %bb35

bb35:                                             ; preds = %bb27
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  %tmp36 = add nuw nsw i32 %.01, 1
  br label %bb25

bb38:                                             ; preds = %bb48, %bb25
  %indvars.iv4 = phi i64 [ %indvars.iv.next5, %bb48 ], [ 0, %bb25 ]
  %.02 = phi i32 [ 0, %bb25 ], [ %tmp49, %bb48 ]
  %exitcond6 = icmp ne i64 %indvars.iv4, 100
  br i1 %exitcond6, label %bb40, label %bb51

bb40:                                             ; preds = %bb38
  %tmp41 = add nsw i32 %.02, -3
  %tmp42 = add nuw nsw i64 %indvars.iv4, 3
  %tmp43 = trunc i64 %tmp42 to i32
  %tmp44 = mul nsw i32 %tmp41, %tmp43
  %tmp45 = trunc i64 %indvars.iv4 to i32
  %tmp46 = srem i32 %tmp44, %tmp45
  %tmp47 = getelementptr inbounds [1024 x i32], [1024 x i32]* @C, i64 0, i64 %indvars.iv4
  store i32 %tmp46, i32* %tmp47, align 4
  br label %bb48

bb48:                                             ; preds = %bb40
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  %tmp49 = add nuw nsw i32 %.02, 1
  br label %bb38

bb51:                                             ; preds = %bb61, %bb38
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb61 ], [ 0, %bb38 ]
  %.03 = phi i32 [ 0, %bb38 ], [ %tmp62, %bb61 ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %bb53, label %bb52

bb52:                                             ; preds = %bb51
  br label %bb63

bb53:                                             ; preds = %bb51
  %tmp54 = add nsw i32 %.03, -3
  %tmp55 = add nuw nsw i64 %indvars.iv, 3
  %tmp56 = trunc i64 %tmp55 to i32
  %tmp57 = mul nsw i32 %tmp54, %tmp56
  %tmp58 = trunc i64 %indvars.iv to i32
  %tmp59 = srem i32 %tmp57, %tmp58
  %tmp60 = getelementptr inbounds [1024 x i32], [1024 x i32]* @D, i64 0, i64 %indvars.iv
  store i32 %tmp59, i32* %tmp60, align 4
  br label %bb61

bb61:                                             ; preds = %bb53
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %tmp62 = add nuw nsw i32 %.03, 1
  br label %bb51

bb63:                                             ; preds = %bb52
  ret void
}
