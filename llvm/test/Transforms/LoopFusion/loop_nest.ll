; RUN: opt -S -passes=loop-fusion < %s | FileCheck %s
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
; CHECK: br label %[[LOOP3HEADER:bb[0-9]+]]
; CHECK: [[LOOP3HEADER]]
; CHECK: br label %[[LOOP2HEADER:bb[0-9]+]]
; CHECK: [[LOOP2HEADER]]
; CHECK: br label %[[LOOP4HEADER:bb[0-9]+]]
; CHECK: [[LOOP4HEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP3HEADER]], label %[[LOOP1LATCH:bb[0-9]+]]
; CHECK: [[LOOP1LATCH]]
; CHECK-NEXT: %inc.outer.fc0 = add nuw nsw i64 %indvars.iv105, 1
; CHECK-NEXT: %add.outer.fc0 = add nuw nsw i32 %.06, 1
; CHECK-NEXT: %cmp.outer.fc0 = icmp ne i64 %inc.outer.fc0, 100
; CHECK: br i1 %{{.*}}, label %[[LOOP1HEADER]], label %[[LOOP1EXIT:bb[0-9]*]]
; CHECK: ret void

; TODO: The current version of loop fusion does not allow the inner loops to be
; fused because they are not control flow equivalent and adjacent. These are
; limitations that can be addressed in future improvements to fusion.
define void @dep_free() {
bb:
  br label %bb16

bb16:                                   ; preds = %bb, %bb27
  %.06 = phi i32 [ 0, %bb ], [ %add.outer.fc0, %bb27 ]
  %indvars.iv105 = phi i64 [ 0, %bb ], [ %inc.outer.fc0, %bb27 ]
  br label %bb18

bb30:                                   ; preds = %bb27
  br label %bb33

bb18:                                             ; preds = %bb16, %bb25
  %indvars.iv74 = phi i64 [ 0, %bb16 ], [ %indvars.iv.next8, %bb25 ]
  %tmp = add nsw i32 %.06, -3
  %tmp19 = add nuw nsw i64 %indvars.iv105, 3
  %tmp20 = trunc i64 %tmp19 to i32
  %tmp21 = mul nsw i32 %tmp, %tmp20
  %tmp22 = trunc i64 %indvars.iv105 to i32
  %tmp23 = srem i32 %tmp21, %tmp22
  %tmp24 = getelementptr inbounds [1024 x [1024 x i32]], ptr @A, i64 0, i64 %indvars.iv105, i64 %indvars.iv74
  store i32 %tmp23, ptr %tmp24, align 4
  br label %bb25

bb25:                                             ; preds = %bb18
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv74, 1
  %exitcond9 = icmp ne i64 %indvars.iv.next8, 100
  br i1 %exitcond9, label %bb18, label %bb27

bb27:                                             ; preds = %bb25
  %inc.outer.fc0 = add nuw nsw i64 %indvars.iv105, 1
  %add.outer.fc0 = add nuw nsw i32 %.06, 1
  %cmp.outer.fc0 = icmp ne i64 %inc.outer.fc0, 100
  br i1 %cmp.outer.fc0, label %bb16, label %bb30

bb33:                                   ; preds = %bb30, %bb45
  %.023 = phi i32 [ 0, %bb30 ], [ %tmp46, %bb45 ]
  %indvars.iv42 = phi i64 [ 0, %bb30 ], [ %indvars.iv.next5, %bb45 ]
  br label %bb35

bb31:                                             ; preds = %bb45
  br label %bb47

bb35:                                             ; preds = %bb33, %bb43
  %indvars.iv1 = phi i64 [ 0, %bb33 ], [ %indvars.iv.next, %bb43 ]
  %tmp36 = add nsw i32 %.023, -3
  %tmp37 = add nuw nsw i64 %indvars.iv42, 3
  %tmp38 = trunc i64 %tmp37 to i32
  %tmp39 = mul nsw i32 %tmp36, %tmp38
  %tmp40 = trunc i64 %indvars.iv42 to i32
  %tmp41 = srem i32 %tmp39, %tmp40
  %tmp42 = getelementptr inbounds [1024 x [1024 x i32]], ptr @B, i64 0, i64 %indvars.iv42, i64 %indvars.iv1
  store i32 %tmp41, ptr %tmp42, align 4
  br label %bb43

bb43:                                             ; preds = %bb35
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %bb35, label %bb45

bb45:                                             ; preds = %bb43
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv42, 1
  %tmp46 = add nuw nsw i32 %.023, 1
  %exitcond6 = icmp ne i64 %indvars.iv.next5, 100
  br i1 %exitcond6, label %bb33, label %bb31

bb47:                                             ; preds = %bb31
  ret void
}
