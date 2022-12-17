; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa 2>&1 | FileCheck -check-prefix=IR %s
; RUN: FileCheck --input-file=%t %s

; Inner loop only reductions are not supported currently. See discussion at
; D53027 for more information on the required checks.

@A = common global [500 x [500 x i32]] zeroinitializer
@X = common global i32 0
@B = common global [500 x [500 x i32]] zeroinitializer
@Y = common global i32 0

;; global X

;;  for( int i=1;i<N;i++)
;;    for( int j=1;j<N;j++)
;;      X+=A[j][i];

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHI
; CHECK-NEXT: Function:        reduction_01

; IR-LABEL: @reduction_01(
; IR-NOT: split

define void @reduction_01(i32 %N) {
entry:
  %cmp16 = icmp sgt i32 %N, 1
  br i1 %cmp16, label %for.body3.lr.ph, label %for.end8

for.body3.lr.ph:                                  ; preds = %for.cond1.for.inc6_crit_edge, %entry
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
  %add.lcssa = phi i32 [ %add, %for.body3 ]
  store i32 %add.lcssa, i32* @X
  %indvars.iv.next19 = add nuw nsw i64 %indvars.iv18, 1
  %lftr.wideiv20 = trunc i64 %indvars.iv.next19 to i32
  %exitcond21 = icmp eq i32 %lftr.wideiv20, %N
  br i1 %exitcond21, label %for.end8, label %for.body3.lr.ph

for.end8:                                         ; preds = %for.cond1.for.inc6_crit_edge, %entry
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

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHIOuter
; CHECK-NEXT: Function:        reduction_03

; IR-LABEL: @reduction_03(
; IR-NOT: split

define void @reduction_03(i32 %N) {
entry:
  %cmp35 = icmp sgt i32 %N, 1
  br i1 %cmp35, label %for.cond4.preheader.lr.ph, label %for.end19

for.cond4.preheader.lr.ph:                        ; preds = %for.cond1.for.inc17_crit_edge, %entry
  %indvars.iv41 = phi i64 [ %indvars.iv.next42, %for.cond1.for.inc17_crit_edge ], [ 1, %entry ]
  %Y.promoted = load i32, i32* @Y
  br label %for.body6.lr.ph

for.body6.lr.ph:                                  ; preds = %for.cond4.for.end_crit_edge, %for.cond4.preheader.lr.ph
  %indvars.iv37 = phi i64 [ 1, %for.cond4.preheader.lr.ph ], [ %indvars.iv.next38, %for.cond4.for.end_crit_edge ]
  %add1334 = phi i32 [ %Y.promoted, %for.cond4.preheader.lr.ph ], [ %add13, %for.cond4.for.end_crit_edge ]
  %X.promoted = load i32, i32* @X
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body6.lr.ph
  %indvars.iv = phi i64 [ 1, %for.body6.lr.ph ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv37
  %0 = load i32, i32* %arrayidx8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.cond4.for.end_crit_edge, label %for.body6

for.cond4.for.end_crit_edge:                      ; preds = %for.body6
  %arrayidx12 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @B, i64 0, i64 %indvars.iv37, i64 %indvars.iv41
  %1 = load i32, i32* %arrayidx12
  %add13 = add nsw i32 %add1334, %1
  %indvars.iv.next38 = add nuw nsw i64 %indvars.iv37, 1
  %lftr.wideiv39 = trunc i64 %indvars.iv.next38 to i32
  %exitcond40 = icmp eq i32 %lftr.wideiv39, %N
  br i1 %exitcond40, label %for.cond1.for.inc17_crit_edge, label %for.body6.lr.ph

for.cond1.for.inc17_crit_edge:                    ; preds = %for.cond4.for.end_crit_edge
  %add13.lcssa = phi i32 [ %add13, %for.cond4.for.end_crit_edge ]
  store i32 %add13.lcssa, i32* @Y
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %lftr.wideiv43 = trunc i64 %indvars.iv.next42 to i32
  %exitcond44 = icmp eq i32 %lftr.wideiv43, %N
  br i1 %exitcond44, label %for.end19, label %for.cond4.preheader.lr.ph

for.end19:                                        ; preds = %for.cond1.for.inc17_crit_edge, %entry
  ret void
}
