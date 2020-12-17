; RUN: opt -S -loop-simplify -loop-fusion -debug-only=loop-fusion -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

@B = common global [1024 x i32] zeroinitializer, align 16

; CHECK that the two candidates for fusion are placed into separate candidate
; sets because they are not control flow equivalent.

; CHECK: Performing Loop Fusion on function non_cfe
; CHECK: Fusion Candidates:
; CHECK: *** Fusion Candidate Set ***
; CHECK: bb
; CHECK: ****************************
; CHECK: *** Fusion Candidate Set ***
; CHECK: bb20.preheader
; CHECK: ****************************
; CHECK: Loop Fusion complete
define void @non_cfe(i32* noalias %arg, i32 %N) {
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
  br i1 %exitcond4, label %bb7, label %bb34

bb34:
  %cmp = icmp slt i32 %N, 50
  br i1 %cmp, label %bb16, label %bb33

bb16:                                             ; preds = %bb34
  %tmp17 = load i32, i32* %arg, align 4
  %tmp18 = icmp slt i32 %tmp17, 0
  br i1 %tmp18, label %bb20.preheader, label %bb33

bb20.preheader:                                   ; preds = %bb16
  br label %bb22

bb22:                                             ; preds = %bb20.preheader, %bb30
  %.02 = phi i32 [ 0, %bb20.preheader ], [ %tmp31, %bb30 ]
  %indvars.iv1 = phi i64 [ 0, %bb20.preheader ], [ %indvars.iv.next, %bb30 ]
  %tmp23 = add nsw i32 %.02, -3
  %tmp24 = add nuw nsw i64 %indvars.iv1, 3
  %tmp25 = trunc i64 %tmp24 to i32
  %tmp26 = mul nsw i32 %tmp23, %tmp25
  %tmp27 = trunc i64 %indvars.iv1 to i32
  %tmp28 = srem i32 %tmp26, %tmp27
  %tmp29 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv1
  store i32 %tmp28, i32* %tmp29, align 4
  br label %bb30

bb30:                                             ; preds = %bb22
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %tmp31 = add nuw nsw i32 %.02, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %bb22, label %bb33.loopexit

bb33.loopexit:                                    ; preds = %bb30
  br label %bb33

bb33:                                             ; preds = %bb33.loopexit, %bb16, %bb34
  ret void
}

; Check that fusion detects the two canddates are not adjacent (the exit block
; of the first candidate is not the preheader of the second candidate).

; CHECK: Performing Loop Fusion on function non_adjacent
; CHECK: Fusion Candidates:
; CHECK: *** Fusion Candidate Set ***
; CHECK-NEXT: [[LOOP1PREHEADER:bb[0-9]*]]
; CHECK-NEXT: [[LOOP2PREHEADER:bb[0-9]*]]
; CHECK-NEXT: ****************************
; CHECK: Attempting fusion on Candidate Set:
; CHECK-NEXT: [[LOOP1PREHEADER]]
; CHECK-NEXT: [[LOOP2PREHEADER]]
; CHECK: Fusion candidates are not adjacent. Not fusing.
; CHECK: Loop Fusion complete
define void @non_adjacent(i32* noalias %arg) {
bb:
  br label %bb5

bb4:                                              ; preds = %bb11
  br label %bb13

bb5:                                              ; preds = %bb, %bb11
  %.013 = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  %tmp = add nsw i64 %.013, -3
  %tmp6 = add nuw nsw i64 %.013, 3
  %tmp7 = mul nsw i64 %tmp, %tmp6
  %tmp8 = srem i64 %tmp7, %.013
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = getelementptr inbounds i32, i32* %arg, i64 %.013
  store i32 %tmp9, i32* %tmp10, align 4
  br label %bb11

bb11:                                             ; preds = %bb5
  %tmp12 = add nuw nsw i64 %.013, 1
  %exitcond2 = icmp ne i64 %tmp12, 100
  br i1 %exitcond2, label %bb5, label %bb4

bb13:                                             ; preds = %bb4
  br label %bb16

bb15:                                             ; preds = %bb23
  br label %bb25

bb16:                                             ; preds = %bb13, %bb23
  %.02 = phi i64 [ 0, %bb13 ], [ %tmp24, %bb23 ]
  %tmp17 = add nsw i64 %.02, -3
  %tmp18 = add nuw nsw i64 %.02, 3
  %tmp19 = mul nsw i64 %tmp17, %tmp18
  %tmp20 = srem i64 %tmp19, %.02
  %tmp21 = trunc i64 %tmp20 to i32
  %tmp22 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %.02
  store i32 %tmp21, i32* %tmp22, align 4
  br label %bb23

bb23:                                             ; preds = %bb16
  %tmp24 = add nuw nsw i64 %.02, 1
  %exitcond = icmp ne i64 %tmp24, 100
  br i1 %exitcond, label %bb16, label %bb15

bb25:                                             ; preds = %bb15
  ret void
}

; Check that the different bounds are detected and prevent fusion.

; CHECK: Performing Loop Fusion on function different_bounds
; CHECK: Fusion Candidates:
; CHECK: *** Fusion Candidate Set ***
; CHECK-NEXT: [[LOOP1PREHEADER:bb[0-9]*]]
; CHECK-NEXT: [[LOOP2PREHEADER:bb[0-9]*]]
; CHECK-NEXT: ****************************
; CHECK: Attempting fusion on Candidate Set:
; CHECK-NEXT: [[LOOP1PREHEADER]]
; CHECK-NEXT: [[LOOP2PREHEADER]]
; CHECK: Fusion candidates do not have identical trip counts. Not fusing.
; CHECK: Loop Fusion complete
define void @different_bounds(i32* noalias %arg) {
bb:
  br label %bb5

bb4:                                              ; preds = %bb11
  br label %bb13

bb5:                                              ; preds = %bb, %bb11
  %.013 = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  %tmp = add nsw i64 %.013, -3
  %tmp6 = add nuw nsw i64 %.013, 3
  %tmp7 = mul nsw i64 %tmp, %tmp6
  %tmp8 = srem i64 %tmp7, %.013
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = getelementptr inbounds i32, i32* %arg, i64 %.013
  store i32 %tmp9, i32* %tmp10, align 4
  br label %bb11

bb11:                                             ; preds = %bb5
  %tmp12 = add nuw nsw i64 %.013, 1
  %exitcond2 = icmp ne i64 %tmp12, 100
  br i1 %exitcond2, label %bb5, label %bb4

bb13:                                             ; preds = %bb4
  br label %bb16

bb15:                                             ; preds = %bb23
  br label %bb25

bb16:                                             ; preds = %bb13, %bb23
  %.02 = phi i64 [ 0, %bb13 ], [ %tmp24, %bb23 ]
  %tmp17 = add nsw i64 %.02, -3
  %tmp18 = add nuw nsw i64 %.02, 3
  %tmp19 = mul nsw i64 %tmp17, %tmp18
  %tmp20 = srem i64 %tmp19, %.02
  %tmp21 = trunc i64 %tmp20 to i32
  %tmp22 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %.02
  store i32 %tmp21, i32* %tmp22, align 4
  br label %bb23

bb23:                                             ; preds = %bb16
  %tmp24 = add nuw nsw i64 %.02, 1
  %exitcond = icmp ne i64 %tmp24, 200
  br i1 %exitcond, label %bb16, label %bb15

bb25:                                             ; preds = %bb15
  ret void
}

; Check that the negative dependence between the two candidates is identified
; and prevents fusion.

; CHECK: Performing Loop Fusion on function negative_dependence
; CHECK: Fusion Candidates:
; CHECK: *** Fusion Candidate Set ***
; CHECK-NEXT: [[LOOP1PREHEADER:bb[0-9]*]]
; CHECK-NEXT: [[LOOP2PREHEADER:bb[0-9]*]]
; CHECK-NEXT: ****************************
; CHECK: Attempting fusion on Candidate Set:
; CHECK-NEXT: [[LOOP1PREHEADER]]
; CHECK-NEXT: [[LOOP2PREHEADER]]
; CHECK: Memory dependencies do not allow fusion!
; CHECK: Loop Fusion complete
define void @negative_dependence(i32* noalias %arg) {
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
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %tmp14 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv.next
  %tmp15 = load i32, i32* %tmp14, align 4
  %tmp16 = shl nsw i32 %tmp15, 1
  %tmp17 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv1
  store i32 %tmp16, i32* %tmp17, align 4
  br label %bb18

bb18:                                             ; preds = %bb13
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %bb13, label %bb19

bb19:                                             ; preds = %bb18
  ret void
}

; Check for values defined in Loop 0 and used in Loop 1.
; It is not safe to fuse in this case, because the second loop has
; a use of %.01.lcssa which is defined in the body of loop 0. The
; first loop must execute completely in order to compute the correct
; value of %.01.lcssa to be used in the second loop.

; CHECK: Performing Loop Fusion on function sumTest
; CHECK: Fusion Candidates:
; CHECK: *** Fusion Candidate Set ***
; CHECK-NEXT: [[LOOP1PREHEADER:bb[0-9]*]]
; CHECK-NEXT: [[LOOP2PREHEADER:bb[0-9]*]]
; CHECK-NEXT: ****************************
; CHECK: Attempting fusion on Candidate Set:
; CHECK-NEXT: [[LOOP1PREHEADER]]
; CHECK-NEXT: [[LOOP2PREHEADER]]
; CHECK: Memory dependencies do not allow fusion!
; CHECK: Loop Fusion complete
define i32 @sumTest(i32* noalias %arg) {
bb:
  br label %bb9

bb13.preheader:                                   ; preds = %bb9
  br label %bb15

bb9:                                              ; preds = %bb, %bb9
  %.01.lcssa = phi i32 [ 0, %bb ], [ %tmp11, %bb9 ]
  %.013 = phi i32 [ 0, %bb ], [ %tmp11, %bb9 ]
  %indvars.iv32 = phi i64 [ 0, %bb ], [ %indvars.iv.next4, %bb9 ]
  %tmp = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv32
  %tmp10 = load i32, i32* %tmp, align 4
  %tmp11 = add nsw i32 %.013, %tmp10
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv32, 1
  %exitcond5 = icmp ne i64 %indvars.iv.next4, 100
  br i1 %exitcond5, label %bb9, label %bb13.preheader

bb14:                                             ; preds = %bb20
  br label %bb21

bb15:                                             ; preds = %bb13.preheader, %bb20
  %indvars.iv1 = phi i64 [ 0, %bb13.preheader ], [ %indvars.iv.next, %bb20 ]
  %tmp16 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv1
  %tmp17 = load i32, i32* %tmp16, align 4
  %tmp18 = sdiv i32 %tmp17, %.01.lcssa
  %tmp19 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv1
  store i32 %tmp18, i32* %tmp19, align 4
  br label %bb20

bb20:                                             ; preds = %bb15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %bb15, label %bb14

bb21:                                             ; preds = %bb14
  ret i32 %.01.lcssa
}

; Similar to sumTest above. The first loop computes %add and must
; complete before it is used in the second loop. Thus, these two loops
; also cannot be fused.

; CHECK: Performing Loop Fusion on function test
; CHECK: Fusion Candidates:
; CHECK: *** Fusion Candidate Set ***
; CHECK-NEXT: [[LOOP1PREHEADER:for.body[0-9]*.preheader]]
; CHECK-NEXT: [[LOOP2PREHEADER:for.body[0-9]*.preheader]]
; CHECK-NEXT: ****************************
; CHECK: Attempting fusion on Candidate Set:
; CHECK-NEXT: [[LOOP1PREHEADER]]
; CHECK-NEXT: [[LOOP2PREHEADER]]
; CHECK: Memory dependencies do not allow fusion!
; CHECK: Loop Fusion complete
define float @test(float* nocapture %a, i32 %n) {
entry:
  %conv = zext i32 %n to i64
  %cmp32 = icmp eq i32 %n, 0
  br i1 %cmp32, label %for.cond.cleanup7, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.034 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum1.033 = phi float [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %idxprom = trunc i64 %i.034 to i32
  %arrayidx = getelementptr inbounds float, float* %a, i32 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %sum1.033, %0
  %inc = add nuw nsw i64 %i.034, 1
  %cmp = icmp ult i64 %inc, %conv
  br i1 %cmp, label %for.body, label %for.body8

for.body8:                                        ; preds = %for.body, %for.body8
  %i2.031 = phi i64 [ %inc14, %for.body8 ], [ 0, %for.body ]
  %idxprom9 = trunc i64 %i2.031 to i32
  %arrayidx10 = getelementptr inbounds float, float* %a, i32 %idxprom9
  %1 = load float, float* %arrayidx10, align 4
  %div = fdiv float %1, %add
  store float %div, float* %arrayidx10, align 4
  %inc14 = add nuw nsw i64 %i2.031, 1
  %cmp5 = icmp ult i64 %inc14, %conv
  br i1 %cmp5, label %for.body8, label %for.cond.cleanup7

for.cond.cleanup7:                                ; preds = %for.body8, %entry
  %sum1.0.lcssa36 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body8 ]
  ret float %sum1.0.lcssa36
}

; Check that non-rotated loops are not considered for fusion.
; CHECK: Performing Loop Fusion on function notRotated
; CHECK: Loop bb{{.*}} is not rotated!
; CHECK: Loop bb{{.*}} is not rotated!
define void @notRotated(i32* noalias %arg) {
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
