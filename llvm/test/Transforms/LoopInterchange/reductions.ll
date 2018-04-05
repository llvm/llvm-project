; REQUIRES: asserts
; RUN: opt < %s -basicaa -loop-interchange -verify-dom-info -verify-loop-info -S -debug 2>&1 | FileCheck %s

@A = common global [500 x [500 x i32]] zeroinitializer
@X = common global i32 0
@B = common global [500 x [500 x i32]] zeroinitializer
@Y = common global i32 0

;;  for( int i=1;i<N;i++)
;;    for( int j=1;j<N;j++)
;;      X+=A[j][i];

;; Loop is interchanged check that the phi nodes are split and the promoted value is used instead of the reduction phi.
; CHECK: Loops interchanged.

define void @reduction_01(i32 %N) {
entry:
  %cmp16 = icmp sgt i32 %N, 1
  br i1 %cmp16, label %for.body3.lr.ph, label %for.end8

for.body3.lr.ph:                                  ; preds = %entry, %for.cond1.for.inc6_crit_edge
  %indvars.iv18 = phi i64 [ %indvars.iv.next19, %for.cond1.for.inc6_crit_edge ], [ 1, %entry ]
  %X.promoted = load i32, i32* @X
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ 1, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %add15 = phi i32 [ %X.promoted, %for.body3.lr.ph ], [ %add, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv18
  %0 = load i32, i32* %arrayidx5
  %add = add nsw i32 %add15, %0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.cond1.for.inc6_crit_edge, label %for.body3

for.cond1.for.inc6_crit_edge:                     ; preds = %for.body3
  store i32 %add, i32* @X
  %indvars.iv.next19 = add nuw nsw i64 %indvars.iv18, 1
  %lftr.wideiv20 = trunc i64 %indvars.iv.next19 to i32
  %exitcond21 = icmp eq i32 %lftr.wideiv20, %N
  br i1 %exitcond21, label %for.end8, label %for.body3.lr.ph

for.end8:                                         ; preds = %for.cond1.for.inc6_crit_edge, %entry
  ret void
}

;; Test for more than 1 reductions inside a loop.
;;  for( int i=1;i<N;i++)
;;    for( int j=1;j<N;j++)
;;      for( int k=1;k<N;k++) {
;;        X+=A[k][j];
;;        Y+=B[k][i];
;;      }

;; Loop is interchanged check that the phi nodes are split and the promoted value is used instead of the reduction phi.
; CHECK: Loops interchanged.

define void @reduction_02(i32 %N)  {
entry:
  %cmp34 = icmp sgt i32 %N, 1
  br i1 %cmp34, label %for.cond4.preheader.preheader, label %for.end19

for.cond4.preheader.preheader:                    ; preds = %entry, %for.inc17
  %indvars.iv40 = phi i64 [ %indvars.iv.next41, %for.inc17 ], [ 1, %entry ]
  br label %for.body6.lr.ph

for.body6.lr.ph:                                  ; preds = %for.cond4.for.inc14_crit_edge, %for.cond4.preheader.preheader
  %indvars.iv36 = phi i64 [ %indvars.iv.next37, %for.cond4.for.inc14_crit_edge ], [ 1, %for.cond4.preheader.preheader ]
  %X.promoted = load i32, i32* @X
  %Y.promoted = load i32, i32* @Y
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body6.lr.ph
  %indvars.iv = phi i64 [ 1, %for.body6.lr.ph ], [ %indvars.iv.next, %for.body6 ]
  %add1331 = phi i32 [ %Y.promoted, %for.body6.lr.ph ], [ %add13, %for.body6 ]
  %add30 = phi i32 [ %X.promoted, %for.body6.lr.ph ], [ %add, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv36
  %0 = load i32, i32* %arrayidx8
  %add = add nsw i32 %add30, %0
  %arrayidx12 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @B, i64 0, i64 %indvars.iv, i64 %indvars.iv40
  %1 = load i32, i32* %arrayidx12
  %add13 = add nsw i32 %add1331, %1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.cond4.for.inc14_crit_edge, label %for.body6

for.cond4.for.inc14_crit_edge:                    ; preds = %for.body6
  store i32 %add, i32* @X
  store i32 %add13, i32* @Y
  %indvars.iv.next37 = add nuw nsw i64 %indvars.iv36, 1
  %lftr.wideiv38 = trunc i64 %indvars.iv.next37 to i32
  %exitcond39 = icmp eq i32 %lftr.wideiv38, %N
  br i1 %exitcond39, label %for.inc17, label %for.body6.lr.ph

for.inc17:                                        ; preds = %for.cond4.for.inc14_crit_edge
  %indvars.iv.next41 = add nuw nsw i64 %indvars.iv40, 1
  %lftr.wideiv42 = trunc i64 %indvars.iv.next41 to i32
  %exitcond43 = icmp eq i32 %lftr.wideiv42, %N
  br i1 %exitcond43, label %for.end19, label %for.cond4.preheader.preheader

for.end19:                                        ; preds = %for.inc17, %entry
  ret void
}

;; Not tightly nested. Do not interchange.
;;  for( int i=1;i<N;i++)
;;    for( int j=1;j<N;j++) {
;;      for( int k=1;k<N;k++) {
;;        X+=A[k][j];
;;      }
;;      Y+=B[j][i];
;;    }

;; Not tightly nested. Do not interchange.
;; Not interchanged hence the phi's in the inner loop will not be split.
; CHECK: Outer loops with reductions are not supported currently.

define void @reduction_03(i32 %N)  {
entry:
  %cmp35 = icmp sgt i32 %N, 1
  br i1 %cmp35, label %for.cond4.preheader.lr.ph, label %for.end19

for.cond4.preheader.lr.ph:                        ; preds = %entry, %for.cond1.for.inc17_crit_edge
  %indvars.iv41 = phi i64 [ %indvars.iv.next42, %for.cond1.for.inc17_crit_edge ], [ 1, %entry ]
  %Y.promoted = load i32, i32* @Y
  br label %for.body6.lr.ph

for.body6.lr.ph:                                  ; preds = %for.cond4.preheader.lr.ph, %for.cond4.for.end_crit_edge
  %indvars.iv37 = phi i64 [ 1, %for.cond4.preheader.lr.ph ], [ %indvars.iv.next38, %for.cond4.for.end_crit_edge ]
  %add1334 = phi i32 [ %Y.promoted, %for.cond4.preheader.lr.ph ], [ %add13, %for.cond4.for.end_crit_edge ]
  %X.promoted = load i32, i32* @X
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body6.lr.ph
  %indvars.iv = phi i64 [ 1, %for.body6.lr.ph ], [ %indvars.iv.next, %for.body6 ]
  %add31 = phi i32 [ %X.promoted, %for.body6.lr.ph ], [ %add, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv37
  %0 = load i32, i32* %arrayidx8
  %add = add nsw i32 %add31, %0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.cond4.for.end_crit_edge, label %for.body6

for.cond4.for.end_crit_edge:                      ; preds = %for.body6
  store i32 %add, i32* @X
  %arrayidx12 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @B, i64 0, i64 %indvars.iv37, i64 %indvars.iv41
  %1 = load i32, i32* %arrayidx12
  %add13 = add nsw i32 %add1334, %1
  %indvars.iv.next38 = add nuw nsw i64 %indvars.iv37, 1
  %lftr.wideiv39 = trunc i64 %indvars.iv.next38 to i32
  %exitcond40 = icmp eq i32 %lftr.wideiv39, %N
  br i1 %exitcond40, label %for.cond1.for.inc17_crit_edge, label %for.body6.lr.ph

for.cond1.for.inc17_crit_edge:                    ; preds = %for.cond4.for.end_crit_edge
  store i32 %add13, i32* @Y
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %lftr.wideiv43 = trunc i64 %indvars.iv.next42 to i32
  %exitcond44 = icmp eq i32 %lftr.wideiv43, %N
  br i1 %exitcond44, label %for.end19, label %for.cond4.preheader.lr.ph

for.end19:                                        ; preds = %for.cond1.for.inc17_crit_edge, %entry
  ret void
}

;; Multiple use of reduction not safe. Do not interchange.
;;  for( int i=1;i<N;i++)
;;    for( int j=1;j<N;j++)
;;      for( int k=1;k<N;k++) {
;;        X+=A[k][j];
;;        Y+=X;
;;      }

;; Not interchanged hence the phi's in the inner loop will not be split.
; CHECK: Only inner loops with induction or reduction PHI nodes are supported currently.

define void @reduction_04(i32 %N) {
entry:
  %cmp28 = icmp sgt i32 %N, 1
  br i1 %cmp28, label %for.cond4.preheader.preheader, label %for.end15

for.cond4.preheader.preheader:                    ; preds = %entry, %for.inc13
  %i.029 = phi i32 [ %inc14, %for.inc13 ], [ 1, %entry ]
  br label %for.body6.lr.ph

for.body6.lr.ph:                                  ; preds = %for.cond4.for.inc10_crit_edge, %for.cond4.preheader.preheader
  %indvars.iv30 = phi i64 [ %indvars.iv.next31, %for.cond4.for.inc10_crit_edge ], [ 1, %for.cond4.preheader.preheader ]
  %X.promoted = load i32, i32* @X
  %Y.promoted = load i32, i32* @Y
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body6.lr.ph
  %indvars.iv = phi i64 [ 1, %for.body6.lr.ph ], [ %indvars.iv.next, %for.body6 ]
  %add925 = phi i32 [ %Y.promoted, %for.body6.lr.ph ], [ %add9, %for.body6 ]
  %add24 = phi i32 [ %X.promoted, %for.body6.lr.ph ], [ %add, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv30
  %0 = load i32, i32* %arrayidx8
  %add = add nsw i32 %add24, %0
  %add9 = add nsw i32 %add925, %add
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.cond4.for.inc10_crit_edge, label %for.body6

for.cond4.for.inc10_crit_edge:                    ; preds = %for.body6
  store i32 %add, i32* @X
  store i32 %add9, i32* @Y
  %indvars.iv.next31 = add nuw nsw i64 %indvars.iv30, 1
  %lftr.wideiv32 = trunc i64 %indvars.iv.next31 to i32
  %exitcond33 = icmp eq i32 %lftr.wideiv32, %N
  br i1 %exitcond33, label %for.inc13, label %for.body6.lr.ph

for.inc13:                                        ; preds = %for.cond4.for.inc10_crit_edge
  %inc14 = add nuw nsw i32 %i.029, 1
  %exitcond34 = icmp eq i32 %inc14, %N
  br i1 %exitcond34, label %for.end15, label %for.cond4.preheader.preheader

for.end15:                                        ; preds = %for.inc13, %entry
  ret void
}
