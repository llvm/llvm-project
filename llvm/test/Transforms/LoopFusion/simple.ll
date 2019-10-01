; RUN: opt -S -loop-fusion < %s | FileCheck %s

@B = common global [1024 x i32] zeroinitializer, align 16

; CHECK: void @dep_free
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1BODY:bb[0-9]*]], label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP1BODY]]
; CHECK: br label %[[LOOP1LATCH:bb[0-9]*]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP2PREHEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP2BODY:bb[0-9]*]], label %[[LOOP2EXIT:bb[0-9]*]]
; CHECK: [[LOOP2BODY]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br label %[[LOOP1HEADER]]
; CHECK: ret void
define void @dep_free(i32* noalias %arg) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb14, %bb
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %bb14 ], [ 0, %bb ]
  %.01 = phi i32 [ 0, %bb ], [ %tmp15, %bb14 ]
  %exitcond4 = icmp ne i64 %indvars.iv2, 100
  br i1 %exitcond4, label %bb7, label %bb17

bb7:                                              ; preds = %bb5
  %tmp = add nsw i32 %.01, -3
  %tmp8 = add nuw nsw i64 %indvars.iv2, 3
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = mul nsw i32 %tmp, %tmp9
  %tmp11 = trunc i64 %indvars.iv2 to i32
  %tmp12 = srem i32 %tmp10, %tmp11
  %tmp13 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv2
  store i32 %tmp12, i32* %tmp13, align 4
  br label %bb14

bb14:                                             ; preds = %bb7
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv2, 1
  %tmp15 = add nuw nsw i32 %.01, 1
  br label %bb5

bb17:                                             ; preds = %bb27, %bb5
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb27 ], [ 0, %bb5 ]
  %.0 = phi i32 [ 0, %bb5 ], [ %tmp28, %bb27 ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %bb19, label %bb18

bb18:                                             ; preds = %bb17
  br label %bb29

bb19:                                             ; preds = %bb17
  %tmp20 = add nsw i32 %.0, -3
  %tmp21 = add nuw nsw i64 %indvars.iv, 3
  %tmp22 = trunc i64 %tmp21 to i32
  %tmp23 = mul nsw i32 %tmp20, %tmp22
  %tmp24 = trunc i64 %indvars.iv to i32
  %tmp25 = srem i32 %tmp23, %tmp24
  %tmp26 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %tmp25, i32* %tmp26, align 4
  br label %bb27

bb27:                                             ; preds = %bb19
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %tmp28 = add nuw nsw i32 %.0, 1
  br label %bb17

bb29:                                             ; preds = %bb18
  ret void
}

; CHECK: void @dep_free_parametric
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1BODY:bb[0-9]*]], label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP1BODY]]
; CHECK: br label %[[LOOP1LATCH:bb[0-9]*]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP2PREHEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP2BODY:bb[0-9]*]], label %[[LOOP2EXIT:bb[0-9]*]]
; CHECK: [[LOOP2BODY]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br label %[[LOOP1HEADER]]
; CHECK: ret void
define void @dep_free_parametric(i32* noalias %arg, i64 %arg2) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb12, %bb
  %.01 = phi i64 [ 0, %bb ], [ %tmp13, %bb12 ]
  %tmp = icmp slt i64 %.01, %arg2
  br i1 %tmp, label %bb5, label %bb15

bb5:                                              ; preds = %bb3
  %tmp6 = add nsw i64 %.01, -3
  %tmp7 = add nuw nsw i64 %.01, 3
  %tmp8 = mul nsw i64 %tmp6, %tmp7
  %tmp9 = srem i64 %tmp8, %.01
  %tmp10 = trunc i64 %tmp9 to i32
  %tmp11 = getelementptr inbounds i32, i32* %arg, i64 %.01
  store i32 %tmp10, i32* %tmp11, align 4
  br label %bb12

bb12:                                             ; preds = %bb5
  %tmp13 = add nuw nsw i64 %.01, 1
  br label %bb3

bb15:                                             ; preds = %bb25, %bb3
  %.0 = phi i64 [ 0, %bb3 ], [ %tmp26, %bb25 ]
  %tmp16 = icmp slt i64 %.0, %arg2
  br i1 %tmp16, label %bb18, label %bb17

bb17:                                             ; preds = %bb15
  br label %bb27

bb18:                                             ; preds = %bb15
  %tmp19 = add nsw i64 %.0, -3
  %tmp20 = add nuw nsw i64 %.0, 3
  %tmp21 = mul nsw i64 %tmp19, %tmp20
  %tmp22 = srem i64 %tmp21, %.0
  %tmp23 = trunc i64 %tmp22 to i32
  %tmp24 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %.0
  store i32 %tmp23, i32* %tmp24, align 4
  br label %bb25

bb25:                                             ; preds = %bb18
  %tmp26 = add nuw nsw i64 %.0, 1
  br label %bb15

bb27:                                             ; preds = %bb17
  ret void
}

; CHECK: void @raw_only
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1BODY:bb[0-9]*]], label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP1BODY]]
; CHECK: br label %[[LOOP1LATCH:bb[0-9]*]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP2PREHEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP2BODY:bb[0-9]*]], label %[[LOOP2EXIT:bb[0-9]*]]
; CHECK: [[LOOP2BODY]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br label %[[LOOP1HEADER]]
; CHECK: ret void
define void @raw_only(i32* noalias %arg) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb9, %bb
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %bb9 ], [ 0, %bb ]
  %exitcond4 = icmp ne i64 %indvars.iv2, 100
  br i1 %exitcond4, label %bb7, label %bb11

bb7:                                              ; preds = %bb5
  %tmp = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv2
  %tmp8 = trunc i64 %indvars.iv2 to i32
  store i32 %tmp8, i32* %tmp, align 4
  br label %bb9

bb9:                                              ; preds = %bb7
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv2, 1
  br label %bb5

bb11:                                             ; preds = %bb18, %bb5
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb18 ], [ 0, %bb5 ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %bb13, label %bb19

bb13:                                             ; preds = %bb11
  %tmp14 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv
  %tmp15 = load i32, i32* %tmp14, align 4
  %tmp16 = shl nsw i32 %tmp15, 1
  %tmp17 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %tmp16, i32* %tmp17, align 4
  br label %bb18

bb18:                                             ; preds = %bb13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb11

bb19:                                             ; preds = %bb11
  ret void
}

; CHECK: void @raw_only_parametric
; CHECK-NEXT: bb:
; CHECK: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1BODY:bb[0-9]*]], label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP1BODY]]
; CHECK: br label %[[LOOP1LATCH:bb[0-9]*]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP2PREHEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP2BODY:bb[0-9]*]], label %[[LOOP2EXIT:bb[0-9]*]]
; CHECK: [[LOOP2BODY]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br label %[[LOOP1HEADER]]
; CHECK: ret void
define void @raw_only_parametric(i32* noalias %arg, i32 %arg4) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb11, %bb
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %bb11 ], [ 0, %bb ]
  %tmp = sext i32 %arg4 to i64
  %tmp6 = icmp slt i64 %indvars.iv2, %tmp
  br i1 %tmp6, label %bb8, label %bb14

bb8:                                              ; preds = %bb5
  %tmp9 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv2
  %tmp10 = trunc i64 %indvars.iv2 to i32
  store i32 %tmp10, i32* %tmp9, align 4
  br label %bb11

bb11:                                             ; preds = %bb8
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv2, 1
  br label %bb5

bb14:                                             ; preds = %bb22, %bb5
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb22 ], [ 0, %bb5 ]
  %tmp13 = sext i32 %arg4 to i64
  %tmp15 = icmp slt i64 %indvars.iv, %tmp13
  br i1 %tmp15, label %bb17, label %bb23

bb17:                                             ; preds = %bb14
  %tmp18 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv
  %tmp19 = load i32, i32* %tmp18, align 4
  %tmp20 = shl nsw i32 %tmp19, 1
  %tmp21 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %tmp20, i32* %tmp21, align 4
  br label %bb22

bb22:                                             ; preds = %bb17
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb14

bb23:                                             ; preds = %bb14
  ret void
}

; CHECK: void @forward_dep
; CHECK-NEXT: bb:
; CHECK: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1BODY:bb[0-9]*]], label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP1BODY]]
; CHECK: br label %[[LOOP1LATCH:bb[0-9]*]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP2PREHEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP2BODY:bb[0-9]*]], label %[[LOOP2EXIT:bb[0-9]*]]
; CHECK: [[LOOP2BODY]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br label %[[LOOP1HEADER]]
; CHECK: ret void
define void @forward_dep(i32* noalias %arg) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb14, %bb
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %bb14 ], [ 0, %bb ]
  %.01 = phi i32 [ 0, %bb ], [ %tmp15, %bb14 ]
  %exitcond4 = icmp ne i64 %indvars.iv2, 100
  br i1 %exitcond4, label %bb7, label %bb17

bb7:                                              ; preds = %bb5
  %tmp = add nsw i32 %.01, -3
  %tmp8 = add nuw nsw i64 %indvars.iv2, 3
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = mul nsw i32 %tmp, %tmp9
  %tmp11 = trunc i64 %indvars.iv2 to i32
  %tmp12 = srem i32 %tmp10, %tmp11
  %tmp13 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv2
  store i32 %tmp12, i32* %tmp13, align 4
  br label %bb14

bb14:                                             ; preds = %bb7
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv2, 1
  %tmp15 = add nuw nsw i32 %.01, 1
  br label %bb5

bb17:                                             ; preds = %bb25, %bb5
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb25 ], [ 0, %bb5 ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %bb19, label %bb26

bb19:                                             ; preds = %bb17
  %tmp20 = add nsw i64 %indvars.iv, -3
  %tmp21 = getelementptr inbounds i32, i32* %arg, i64 %tmp20
  %tmp22 = load i32, i32* %tmp21, align 4
  %tmp23 = mul nsw i32 %tmp22, 3
  %tmp24 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv
  store i32 %tmp23, i32* %tmp24, align 4
  br label %bb25

bb25:                                             ; preds = %bb19
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb17

bb26:                                             ; preds = %bb17
  ret void
}
