; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@C = common global [100 x [100 x i32]] zeroinitializer

;; FIXME:
;; Test for interchange when we have an lcssa phi. This should ideally be interchanged but it is currently not supported.
;;     for(gi=1;gi<N;gi++)
;;       for(gj=1;gj<M;gj++)
;;         A[gj][gi] = A[gj - 1][gi] + C[gj][gi];

@gi = common global i32 0
@gj = common global i32 0

define void @interchange_07(i32 %N, i32 %M){
entry:
  store i32 1, i32* @gi
  %cmp21 = icmp sgt i32 %N, 1
  br i1 %cmp21, label %for.cond1.preheader.lr.ph, label %for.end16

for.cond1.preheader.lr.ph:
  %cmp218 = icmp sgt i32 %M, 1
  %gi.promoted = load i32, i32* @gi
  %0 = add i32 %M, -1
  %1 = sext i32 %gi.promoted to i64
  %2 = sext i32 %N to i64
  %3 = add i32 %gi.promoted, 1
  %4 = icmp slt i32 %3, %N
  %smax = select i1 %4, i32 %N, i32 %3
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv25 = phi i64 [ %1, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next26, %for.inc14 ]
  br i1 %cmp218, label %for.body3, label %for.inc14

for.body3:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 1, %for.cond1.preheader ]
  %5 = add nsw i64 %indvars.iv, -1
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %5, i64 %indvars.iv25
  %6 = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %indvars.iv, i64 %indvars.iv25
  %7 = load i32, i32* %arrayidx9
  %add = add nsw i32 %7, %6
  %arrayidx13 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv25
  store i32 %add, i32* %arrayidx13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc14, label %for.body3

for.inc14:
  %inc.lcssa23 = phi i32 [ 1, %for.cond1.preheader ], [ %M, %for.body3 ]
  %indvars.iv.next26 = add nsw i64 %indvars.iv25, 1
  %cmp = icmp slt i64 %indvars.iv.next26, %2
  br i1 %cmp, label %for.cond1.preheader, label %for.cond.for.end16_crit_edge

for.cond.for.end16_crit_edge:
  store i32 %inc.lcssa23, i32* @gj
  store i32 %smax, i32* @gi
  br label %for.end16

for.end16:
  ret void
}

; CHECK-LABEL: @interchange_07
; CHECK: for.body3:                                        ; preds = %for.body3.preheader, %for.body3
; CHECK:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 1, %for.body3.preheader ]
; CHECK:   %5 = add nsw i64 %indvars.iv, -1
; CHECK:   %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %5, i64 %indvars.iv25
; CHECK:   %6 = load i32, i32* %arrayidx5
; CHECK:   %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %indvars.iv, i64 %indvars.iv25
