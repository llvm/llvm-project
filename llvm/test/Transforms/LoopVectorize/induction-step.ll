; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=8 -S | FileCheck %s

; int int_inc;
;
;int induction_with_global(int init, int *restrict A, int N) {
;  int x = init;
;  for (int i=0;i<N;i++){
;    A[i] = x;
;    x += int_inc;
;  }
;  return x;
;}

; CHECK-LABEL: @induction_with_global(
; CHECK: %[[INT_INC:.*]] = load i32, i32* @int_inc, align 4
; CHECK: vector.body:
; CHECK:  %[[VAR1:.*]] = insertelement <8 x i32> undef, i32 %[[INT_INC]], i32 0
; CHECK:  %[[VAR2:.*]] = shufflevector <8 x i32> %[[VAR1]], <8 x i32> undef, <8 x i32> zeroinitializer
; CHECK:  mul <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %[[VAR2]]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"


@int_inc = common global i32 0, align 4

define i32 @induction_with_global(i32 %init, i32* noalias nocapture %A, i32 %N) {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %0 = load i32, i32* @int_inc, align 4
  %1 = mul i32 %0, %N
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %x.05 = phi i32 [ %init, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %x.05, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %x.05
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  %2 = add i32 %1, %init
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %x.0.lcssa = phi i32 [ %init, %entry ], [ %2, %for.end.loopexit ]
  ret i32 %x.0.lcssa
}


;int induction_with_loop_inv(int init, int *restrict A, int N, int M) {
;  int x = init;
;  for (int j = 0; j < M; j++) {
;    for (int i=0; i<N; i++){
;      A[i] = x;
;      x += j; // induction step is a loop invariant variable
;    }
;  }
;  return x;
;}

; CHECK-LABEL: @induction_with_loop_inv(
; CHECK: for.cond1.preheader:                            
; CHECK: %[[INDVAR0:.*]] = phi i32 [ 0,
; CHECK: %[[INDVAR1:.*]] = phi i32 [ 0,
; CHECK: vector.body:
; CHECK:  %[[VAR1:.*]] = insertelement <8 x i32> undef, i32 %[[INDVAR1]], i32 0
; CHECK:  %[[VAR2:.*]] = shufflevector <8 x i32> %[[VAR1]], <8 x i32> undef, <8 x i32> zeroinitializer
; CHECK:  mul <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %[[VAR2]]

define i32 @induction_with_loop_inv(i32 %init, i32* noalias nocapture %A, i32 %N, i32 %M) {
entry:
  %cmp10 = icmp sgt i32 %M, 0
  br i1 %cmp10, label %for.cond1.preheader.lr.ph, label %for.end6

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp27 = icmp sgt i32 %N, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc4, %for.cond1.preheader.lr.ph
  %indvars.iv15 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next16, %for.inc4 ]
  %j.012 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc5, %for.inc4 ]
  %x.011 = phi i32 [ %init, %for.cond1.preheader.lr.ph ], [ %x.1.lcssa, %for.inc4 ]
  br i1 %cmp27, label %for.body3.preheader, label %for.inc4

for.body3.preheader:                              ; preds = %for.cond1.preheader
  br label %for.body3

for.body3:                                        ; preds = %for.body3.preheader, %for.body3
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 0, %for.body3.preheader ]
  %x.18 = phi i32 [ %add, %for.body3 ], [ %x.011, %for.body3.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %x.18, i32* %arrayidx, align 4
  %add = add nsw i32 %x.18, %j.012
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.inc4.loopexit, label %for.body3

for.inc4.loopexit:                                ; preds = %for.body3
  %0 = add i32 %x.011, %indvars.iv15
  br label %for.inc4

for.inc4:                                         ; preds = %for.inc4.loopexit, %for.cond1.preheader
  %x.1.lcssa = phi i32 [ %x.011, %for.cond1.preheader ], [ %0, %for.inc4.loopexit ]
  %inc5 = add nuw nsw i32 %j.012, 1
  %indvars.iv.next16 = add i32 %indvars.iv15, %N
  %exitcond17 = icmp eq i32 %inc5, %M
  br i1 %exitcond17, label %for.end6.loopexit, label %for.cond1.preheader

for.end6.loopexit:                                ; preds = %for.inc4
  %x.1.lcssa.lcssa = phi i32 [ %x.1.lcssa, %for.inc4 ]
  br label %for.end6

for.end6:                                         ; preds = %for.end6.loopexit, %entry
  %x.0.lcssa = phi i32 [ %init, %entry ], [ %x.1.lcssa.lcssa, %for.end6.loopexit ]
  ret i32 %x.0.lcssa
}
