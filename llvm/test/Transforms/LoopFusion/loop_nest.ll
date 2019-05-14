; RUN: opt -S -loop-fusion < %s | FileCheck %s
;
;    int A[1024][1024];
;    int B[1024][1024];
;
;    #define EXPENSIVE_PURE_COMPUTATION(i) ((i - 3) * (i + 3) % i)
;
;    void dep_free() {
;
;      for (int i = 0; i < 100; i++)
;        for (int j = 0; j < 100; j++)
;          A[i][j] = EXPENSIVE_PURE_COMPUTATION(i);
;
;      for (int i = 0; i < 100; i++)
;        for (int j = 0; j < 100; j++)
;          B[i][j] = EXPENSIVE_PURE_COMPUTATION(i);
;    }
;
@A = common global [1024 x [1024 x i32]] zeroinitializer, align 16
@B = common global [1024 x [1024 x i32]] zeroinitializer, align 16

; CHECK: void @dep_free
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]+]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %exitcond12, label %[[LOOP3PREHEADER:bb[0-9]+.preheader]], label %[[LOOP2HEADER:bb[0-9]+]]
; CHECK: [[LOOP3PREHEADER]]
; CHECK: br label %[[LOOP3HEADER:bb[0-9]+]]
; CHECK: [[LOOP3HEADER]]
; CHECK: br i1 %exitcond9, label %[[LOOP3BODY:bb[0-9]+]], label %[[LOOP1LATCH:bb[0-9]+]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br label %[[LOOP2HEADER:bb[0-9]+]]
; CHECK: [[LOOP2HEADER]]
; CHECK: br i1 %exitcond6, label %[[LOOP4PREHEADER:bb[0-9]+.preheader]], label %[[LOOP2EXITBLOCK:bb[0-9]+]]
; CHECK: [[LOOP4PREHEADER]]
; CHECK: br label %[[LOOP4HEADER:bb[0-9]+]]
; CHECK: [[LOOP2EXITBLOCK]]
; CHECK-NEXT: br label %[[FUNCEXIT:bb[0-9]+]]
; CHECK: [[LOOP4HEADER]]
; CHECK: br i1 %exitcond, label %[[LOOP4BODY:bb[0-9]+]], label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br label %[[LOOP1HEADER:bb[0-9]+]]
; CHECK: [[FUNCEXIT]]
; CHECK: ret void

; TODO: The current version of loop fusion does not allow the inner loops to be
; fused because they are not control flow equivalent and adjacent. These are
; limitations that can be addressed in future improvements to fusion.
define void @dep_free() {
bb:
  br label %bb13

bb13:                                             ; preds = %bb27, %bb
  %indvars.iv10 = phi i64 [ %indvars.iv.next11, %bb27 ], [ 0, %bb ]
  %.0 = phi i32 [ 0, %bb ], [ %tmp28, %bb27 ]
  %exitcond12 = icmp ne i64 %indvars.iv10, 100
  br i1 %exitcond12, label %bb16, label %bb30

bb16:                                             ; preds = %bb25, %bb13
  %indvars.iv7 = phi i64 [ %indvars.iv.next8, %bb25 ], [ 0, %bb13 ]
  %exitcond9 = icmp ne i64 %indvars.iv7, 100
  br i1 %exitcond9, label %bb18, label %bb27

bb18:                                             ; preds = %bb16
  %tmp = add nsw i32 %.0, -3
  %tmp19 = add nuw nsw i64 %indvars.iv10, 3
  %tmp20 = trunc i64 %tmp19 to i32
  %tmp21 = mul nsw i32 %tmp, %tmp20
  %tmp22 = trunc i64 %indvars.iv10 to i32
  %tmp23 = srem i32 %tmp21, %tmp22
  %tmp24 = getelementptr inbounds [1024 x [1024 x i32]], [1024 x [1024 x i32]]* @A, i64 0, i64 %indvars.iv10, i64 %indvars.iv7
  store i32 %tmp23, i32* %tmp24, align 4
  br label %bb25

bb25:                                             ; preds = %bb18
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  br label %bb16

bb27:                                             ; preds = %bb16
  %indvars.iv.next11 = add nuw nsw i64 %indvars.iv10, 1
  %tmp28 = add nuw nsw i32 %.0, 1
  br label %bb13

bb30:                                             ; preds = %bb45, %bb13
  %indvars.iv4 = phi i64 [ %indvars.iv.next5, %bb45 ], [ 0, %bb13 ]
  %.02 = phi i32 [ 0, %bb13 ], [ %tmp46, %bb45 ]
  %exitcond6 = icmp ne i64 %indvars.iv4, 100
  br i1 %exitcond6, label %bb33, label %bb31

bb31:                                             ; preds = %bb30
  br label %bb47

bb33:                                             ; preds = %bb43, %bb30
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb43 ], [ 0, %bb30 ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %bb35, label %bb45

bb35:                                             ; preds = %bb33
  %tmp36 = add nsw i32 %.02, -3
  %tmp37 = add nuw nsw i64 %indvars.iv4, 3
  %tmp38 = trunc i64 %tmp37 to i32
  %tmp39 = mul nsw i32 %tmp36, %tmp38
  %tmp40 = trunc i64 %indvars.iv4 to i32
  %tmp41 = srem i32 %tmp39, %tmp40
  %tmp42 = getelementptr inbounds [1024 x [1024 x i32]], [1024 x [1024 x i32]]* @B, i64 0, i64 %indvars.iv4, i64 %indvars.iv
  store i32 %tmp41, i32* %tmp42, align 4
  br label %bb43

bb43:                                             ; preds = %bb35
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb33

bb45:                                             ; preds = %bb33
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  %tmp46 = add nuw nsw i32 %.02, 1
  br label %bb30

bb47:                                             ; preds = %bb31
  ret void
}
