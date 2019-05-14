; RUN: opt -S -loop-fusion -debug-only=loop-fusion -disable-output < %s 2>&1 | FileCheck %s
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
define void @non_cfe(i32* noalias %arg) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb14, %bb
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %bb14 ], [ 0, %bb ]
  %.01 = phi i32 [ 0, %bb ], [ %tmp15, %bb14 ]
  %exitcond4 = icmp ne i64 %indvars.iv2, 100
  br i1 %exitcond4, label %bb7, label %bb16

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

bb16:                                             ; preds = %bb5
  %tmp17 = load i32, i32* %arg, align 4
  %tmp18 = icmp slt i32 %tmp17, 0
  br i1 %tmp18, label %bb20, label %bb33

bb20:                                             ; preds = %bb30, %bb16
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb30 ], [ 0, %bb16 ]
  %.0 = phi i32 [ 0, %bb16 ], [ %tmp31, %bb30 ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %bb22, label %bb33

bb22:                                             ; preds = %bb20
  %tmp23 = add nsw i32 %.0, -3
  %tmp24 = add nuw nsw i64 %indvars.iv, 3
  %tmp25 = trunc i64 %tmp24 to i32
  %tmp26 = mul nsw i32 %tmp23, %tmp25
  %tmp27 = trunc i64 %indvars.iv to i32
  %tmp28 = srem i32 %tmp26, %tmp27
  %tmp29 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %tmp28, i32* %tmp29, align 4
  br label %bb30

bb30:                                             ; preds = %bb22
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %tmp31 = add nuw nsw i32 %.0, 1
  br label %bb20

bb33:                                             ; preds = %bb20, %bb16
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
  br label %bb3

bb3:                                              ; preds = %bb11, %bb
  %.01 = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  %exitcond2 = icmp ne i64 %.01, 100
  br i1 %exitcond2, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  br label %bb13

bb5:                                              ; preds = %bb3
  %tmp = add nsw i64 %.01, -3
  %tmp6 = add nuw nsw i64 %.01, 3
  %tmp7 = mul nsw i64 %tmp, %tmp6
  %tmp8 = srem i64 %tmp7, %.01
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = getelementptr inbounds i32, i32* %arg, i64 %.01
  store i32 %tmp9, i32* %tmp10, align 4
  br label %bb11

bb11:                                             ; preds = %bb5
  %tmp12 = add nuw nsw i64 %.01, 1
  br label %bb3

bb13:                                             ; preds = %bb4
  br label %bb14

bb14:                                             ; preds = %bb23, %bb13
  %.0 = phi i64 [ 0, %bb13 ], [ %tmp24, %bb23 ]
  %exitcond = icmp ne i64 %.0, 100
  br i1 %exitcond, label %bb16, label %bb15

bb15:                                             ; preds = %bb14
  br label %bb25

bb16:                                             ; preds = %bb14
  %tmp17 = add nsw i64 %.0, -3
  %tmp18 = add nuw nsw i64 %.0, 3
  %tmp19 = mul nsw i64 %tmp17, %tmp18
  %tmp20 = srem i64 %tmp19, %.0
  %tmp21 = trunc i64 %tmp20 to i32
  %tmp22 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %.0
  store i32 %tmp21, i32* %tmp22, align 4
  br label %bb23

bb23:                                             ; preds = %bb16
  %tmp24 = add nuw nsw i64 %.0, 1
  br label %bb14

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
  br label %bb3

bb3:                                              ; preds = %bb11, %bb
  %.01 = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  %exitcond2 = icmp ne i64 %.01, 100
  br i1 %exitcond2, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  br label %bb13

bb5:                                              ; preds = %bb3
  %tmp = add nsw i64 %.01, -3
  %tmp6 = add nuw nsw i64 %.01, 3
  %tmp7 = mul nsw i64 %tmp, %tmp6
  %tmp8 = srem i64 %tmp7, %.01
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = getelementptr inbounds i32, i32* %arg, i64 %.01
  store i32 %tmp9, i32* %tmp10, align 4
  br label %bb11

bb11:                                             ; preds = %bb5
  %tmp12 = add nuw nsw i64 %.01, 1
  br label %bb3

bb13:                                             ; preds = %bb4
  br label %bb14

bb14:                                             ; preds = %bb23, %bb13
  %.0 = phi i64 [ 0, %bb13 ], [ %tmp24, %bb23 ]
  %exitcond = icmp ne i64 %.0, 200
  br i1 %exitcond, label %bb16, label %bb15

bb15:                                             ; preds = %bb14
  br label %bb25

bb16:                                             ; preds = %bb14
  %tmp17 = add nsw i64 %.0, -3
  %tmp18 = add nuw nsw i64 %.0, 3
  %tmp19 = mul nsw i64 %tmp17, %tmp18
  %tmp20 = srem i64 %tmp19, %.0
  %tmp21 = trunc i64 %tmp20 to i32
  %tmp22 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %.0
  store i32 %tmp21, i32* %tmp22, align 4
  br label %bb23

bb23:                                             ; preds = %bb16
  %tmp24 = add nuw nsw i64 %.0, 1
  br label %bb14

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
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %tmp14 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv.next
  %tmp15 = load i32, i32* %tmp14, align 4
  %tmp16 = shl nsw i32 %tmp15, 1
  %tmp17 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %tmp16, i32* %tmp17, align 4
  br label %bb18

bb18:                                             ; preds = %bb13
  br label %bb11

bb19:                                             ; preds = %bb11
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
  br label %bb6

bb6:                                              ; preds = %bb9, %bb
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %bb9 ], [ 0, %bb ]
  %.01 = phi i32 [ 0, %bb ], [ %tmp11, %bb9 ]
  %exitcond5 = icmp ne i64 %indvars.iv3, 100
  br i1 %exitcond5, label %bb9, label %bb13

bb9:                                              ; preds = %bb6
  %tmp = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv3
  %tmp10 = load i32, i32* %tmp, align 4
  %tmp11 = add nsw i32 %.01, %tmp10
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %bb6

bb13:                                             ; preds = %bb20, %bb6
  %.01.lcssa = phi i32 [ %.01, %bb6 ], [ %.01.lcssa, %bb20 ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb20 ], [ 0, %bb6 ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %bb15, label %bb14

bb14:                                             ; preds = %bb13
  br label %bb21

bb15:                                             ; preds = %bb13
  %tmp16 = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv
  %tmp17 = load i32, i32* %tmp16, align 4
  %tmp18 = sdiv i32 %tmp17, %.01.lcssa
  %tmp19 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %tmp18, i32* %tmp19, align 4
  br label %bb20

bb20:                                             ; preds = %bb15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb13

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
