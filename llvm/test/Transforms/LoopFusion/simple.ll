; RUN: opt -S -loop-fusion < %s | FileCheck %s

@B = common global [1024 x i32] zeroinitializer, align 16

; CHECK: void @dep_free
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br label %[[LOOP2HEADER:bb[0-9]*]]
; CHECK: [[LOOP2HEADER]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1HEADER]], label %{{.*}}
; CHECK: ret void
define void @dep_free(i32* noalias %arg) {
bb:
  br label %bb7

bb7:                                              ; preds = %bb, %bb14
  %.014 = phi i32 [ 0, %bb ], [ %tmp15, %bb14 ]
  %indvars.iv23 = phi i64 [ 0, %bb ], [ %indvars.iv.next3, %bb14 ]
  %tmp = add nsw i32 %.014, -3
  %tmp8 = add nuw nsw i64 %indvars.iv23, 3
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = mul nsw i32 %tmp, %tmp9
  %tmp11 = trunc i64 %indvars.iv23 to i32
  %tmp12 = srem i32 %tmp10, %tmp11
  %tmp13 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv23
  store i32 %tmp12, i32* %tmp13, align 4
  br label %bb14

bb14:                                             ; preds = %bb7
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv23, 1
  %tmp15 = add nuw nsw i32 %.014, 1
  %exitcond4 = icmp ne i64 %indvars.iv.next3, 100
  br i1 %exitcond4, label %bb7, label %bb17.preheader

bb17.preheader:                                   ; preds = %bb14
  br label %bb19

bb19:                                             ; preds = %bb17.preheader, %bb27
  %.02 = phi i32 [ 0, %bb17.preheader ], [ %tmp28, %bb27 ]
  %indvars.iv1 = phi i64 [ 0, %bb17.preheader ], [ %indvars.iv.next, %bb27 ]
  %tmp20 = add nsw i32 %.02, -3
  %tmp21 = add nuw nsw i64 %indvars.iv1, 3
  %tmp22 = trunc i64 %tmp21 to i32
  %tmp23 = mul nsw i32 %tmp20, %tmp22
  %tmp24 = trunc i64 %indvars.iv1 to i32
  %tmp25 = srem i32 %tmp23, %tmp24
  %tmp26 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv1
  store i32 %tmp25, i32* %tmp26, align 4
  br label %bb27

bb27:                                             ; preds = %bb19
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %tmp28 = add nuw nsw i32 %.02, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %bb19, label %bb18

bb18:                                             ; preds = %bb27
  br label %bb29

bb29:                                             ; preds = %bb18
  ret void
}

; CHECK: void @dep_free_parametric
; CHECK-NEXT: bb:
; CHECK: br i1 %{{.*}}, label %[[LOOP1PREHEADER:bb[0-9.a-z]*]], label %[[EXITBLOCK:bb[0-9]*]]
; CHECK: [[LOOP1PREHEADER]]
; CHECK: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br label %[[LOOP2HEADER:bb[0-9]*]]
; CHECK: [[LOOP2HEADER]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1HEADER]], label %[[EXITBLOCK]]
; CHECK: ret void
define void @dep_free_parametric(i32* noalias %arg, i64 %arg2) {
bb:
  %tmp3 = icmp slt i64 0, %arg2
  br i1 %tmp3, label %bb5, label %bb15.preheader

bb5:                                              ; preds = %bb5, %bb12
  %.014 = phi i64 [ 0, %bb ], [ %tmp13, %bb12 ]
  %tmp6 = add nsw i64 %.014, -3
  %tmp7 = add nuw nsw i64 %.014, 3
  %tmp8 = mul nsw i64 %tmp6, %tmp7
  %tmp9 = srem i64 %tmp8, %.014
  %tmp10 = trunc i64 %tmp9 to i32
  %tmp11 = getelementptr inbounds i32, i32* %arg, i64 %.014
  store i32 %tmp10, i32* %tmp11, align 4
  br label %bb12

bb12:                                             ; preds = %bb5
  %tmp13 = add nuw nsw i64 %.014, 1
  %tmp = icmp slt i64 %tmp13, %arg2
  br i1 %tmp, label %bb5, label %bb15.preheader

bb15.preheader:                                   ; preds = %bb12, %bb
  %tmp161 = icmp slt i64 0, %arg2
  br i1 %tmp161, label %bb18, label %bb27

bb18:                                             ; preds = %bb15.preheader, %bb25
  %.02 = phi i64 [ 0, %bb15.preheader ], [ %tmp26, %bb25 ]
  %tmp19 = add nsw i64 %.02, -3
  %tmp20 = add nuw nsw i64 %.02, 3
  %tmp21 = mul nsw i64 %tmp19, %tmp20
  %tmp22 = srem i64 %tmp21, %.02
  %tmp23 = trunc i64 %tmp22 to i32
  %tmp24 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %.02
  store i32 %tmp23, i32* %tmp24, align 4
  br label %bb25

bb25:                                             ; preds = %bb18
  %tmp26 = add nuw nsw i64 %.02, 1
  %tmp16 = icmp slt i64 %tmp26, %arg2
  br i1 %tmp16, label %bb18, label %bb27

bb27:                                             ; preds = %bb17
  ret void
}

; CHECK: void @raw_only
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br label %[[LOOP2HEADER:bb[0-9]*]]
; CHECK: [[LOOP2HEADER]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1HEADER]], label %{{.*}}
; CHECK: ret void
define void @raw_only(i32* noalias %arg) {
bb:
  br label %bb7

bb11.preheader:                                   ; preds = %bb9
  br label %bb13

bb7:                                              ; preds = %bb, %bb9
  %indvars.iv22 = phi i64 [ 0, %bb ], [ %indvars.iv.next3, %bb9 ]
  %tmp = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv22
  %tmp8 = trunc i64 %indvars.iv22 to i32
  store i32 %tmp8, i32* %tmp, align 4
  br label %bb9

bb9:                                              ; preds = %bb7
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv22, 1
  %exitcond4 = icmp ne i64 %indvars.iv.next3, 100
  br i1 %exitcond4, label %bb7, label %bb11.preheader

bb13:                                             ; preds = %bb11.preheader, %bb18
  %indvars.iv1 = phi i64 [ 0, %bb11.preheader ], [ %indvars.iv.next, %bb18 ]
  %tmp14 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv1
  %tmp15 = load i32, i32* %tmp14, align 4
  %tmp16 = shl nsw i32 %tmp15, 1
  %tmp17 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv1
  store i32 %tmp16, i32* %tmp17, align 4
  br label %bb18

bb18:                                             ; preds = %bb13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100 br i1 %exitcond, label %bb13, label %bb19

bb19:                                             ; preds = %bb18
  ret void
}

; CHECK: void @raw_only_parametric
; CHECK-NEXT: bb:
; CHECK: br i1 %{{.*}}, label %[[LOOP1PREHEADER:bb[0-9.a-z]*]], label %[[EXITBLOCK:bb[0-9]*]]
; CHECK: [[LOOP1PREHEADER]]
; CHECK: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1HEADER]], label %[[EXITBLOCK]]
; CHECK: ret void
define void @raw_only_parametric(i32* noalias %arg, i32 %arg4) {
bb:
  %tmp = sext i32 %arg4 to i64
  %tmp64 = icmp sgt i32 %arg4, 0
  br i1 %tmp64, label %bb8, label %bb23

bb8:                                              ; preds = %bb, %bb8
  %indvars.iv25 = phi i64 [ %indvars.iv.next3, %bb8 ], [ 0, %bb ]
  %tmp9 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv25
  %tmp10 = trunc i64 %indvars.iv25 to i32
  store i32 %tmp10, i32* %tmp9, align 4
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv25, 1
  %tmp6 = icmp slt i64 %indvars.iv.next3, %tmp
  br i1 %tmp6, label %bb8, label %bb17

bb17:                                             ; preds = %bb8, %bb17
  %indvars.iv3 = phi i64 [ %indvars.iv.next, %bb17 ], [ 0, %bb8 ]
  %tmp18 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv3
  %tmp19 = load i32, i32* %tmp18, align 4
  %tmp20 = shl nsw i32 %tmp19, 1
  %tmp21 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv3
  store i32 %tmp20, i32* %tmp21, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv3, 1
  %tmp15 = icmp slt i64 %indvars.iv.next, %tmp
  br i1 %tmp15, label %bb17, label %bb23

bb23:                                             ; preds = %bb17, %bb
  ret void
}

; CHECK: void @forward_dep
; CHECK-NEXT: bb:
; CHECK: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br label %[[LOOP2HEADER:bb[0-9]*]]
; CHECK: [[LOOP2HEADER]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1HEADER]], label %{{.*}}
; CHECK: ret void
define void @forward_dep(i32* noalias %arg) {
bb:
  br label %bb7

bb7:                                              ; preds = %bb, %bb14
  %.013 = phi i32 [ 0, %bb ], [ %tmp15, %bb14 ]
  %indvars.iv22 = phi i64 [ 0, %bb ], [ %indvars.iv.next3, %bb14 ]
  %tmp = add nsw i32 %.013, -3
  %tmp8 = add nuw nsw i64 %indvars.iv22, 3
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = mul nsw i32 %tmp, %tmp9
  %tmp11 = trunc i64 %indvars.iv22 to i32
  %tmp12 = srem i32 %tmp10, %tmp11
  %tmp13 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv22
  store i32 %tmp12, i32* %tmp13, align 4
  br label %bb14

bb14:                                             ; preds = %bb7
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv22, 1
  %tmp15 = add nuw nsw i32 %.013, 1
  %exitcond4 = icmp ne i64 %indvars.iv.next3, 100
  br i1 %exitcond4, label %bb7, label %bb19

bb19:                                             ; preds = %bb14, %bb25
  %indvars.iv1 = phi i64 [ 0, %bb14 ], [ %indvars.iv.next, %bb25 ]
  %tmp20 = add nsw i64 %indvars.iv1, -3
  %tmp21 = getelementptr inbounds i32, i32* %arg, i64 %tmp20
  %tmp22 = load i32, i32* %tmp21, align 4
  %tmp23 = mul nsw i32 %tmp22, 3
  %tmp24 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv1
  store i32 %tmp23, i32* %tmp24, align 4
  br label %bb25

bb25:                                             ; preds = %bb19
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %bb19, label %bb26

bb26:                                             ; preds = %bb25
  ret void
}

; Test that instructions in loop 1 latch are moved to the beginning of loop 2
; latch iff it is proven safe. %inc.first and %cmp.first are moved, but
; `store i32 0, i32* %Ai.first` is not.

; CHECK: void @flow_dep
; CHECK-LABEL: entry:
; CHECK-NEXT: br label %for.first
; CHECK-LABEL: for.first:
; CHECK: store i32 0, i32* %Ai.first
; CHECK: %Ai.second =
; CHECK: br label %for.second.latch
; CHECK-LABEL: for.second.latch:
; CHECK-NEXT: %inc.first = add nsw i64 %i.first, 1
; CHECK-NEXT: %cmp.first = icmp slt i64 %inc.first, 100
; CHECK: br i1 %cmp.second, label %for.first, label %for.end
; CHECK-LABEL: for.end:
; CHECK-NEXT: ret void

define void @flow_dep(i32* noalias %A, i32* noalias %B) {
entry:
  br label %for.first

for.first:
  %i.first = phi i64 [ 0, %entry ], [ %inc.first, %for.first ]
  %Ai.first = getelementptr inbounds i32, i32* %A, i64 %i.first
  store i32 0, i32* %Ai.first, align 4
  %inc.first = add nsw i64 %i.first, 1
  %cmp.first = icmp slt i64 %inc.first, 100
  br i1 %cmp.first, label %for.first, label %for.second.preheader

for.second.preheader:
  br label %for.second

for.second:
  %i.second = phi i64 [ %inc.second, %for.second.latch ], [ 0, %for.second.preheader ]
  %Ai.second = getelementptr inbounds i32, i32* %A, i64 %i.second
  %0 = load i32, i32* %Ai.second, align 4
  %Bi = getelementptr inbounds i32, i32* %B, i64 %i.second
  store i32 %0, i32* %Bi, align 4
  br label %for.second.latch

for.second.latch:
  %inc.second = add nsw i64 %i.second, 1
  %cmp.second = icmp slt i64 %inc.second, 100
  br i1 %cmp.second, label %for.second, label %for.end

for.end:
  ret void
}
