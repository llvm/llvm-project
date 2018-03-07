; REQUIRES: asserts
; RUN: opt < %s -basicaa -loop-interchange -verify-dom-info -S -debug 2>&1 | FileCheck %s
;; We test profitability model in these test cases.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x [100 x i32]] zeroinitializer

;;---------------------------------------Test case 01---------------------------------
;; Loops interchange will result in code vectorization and hence profitable. Check for interchange.
;;   for(int i=1;i<N;i++)
;;     for(int j=1;j<N;j++)
;;       A[j][i] = A[j - 1][i] + B[j][i];

; CHECK: Not interchanging loops. Cannot prove legality.

define void @interchange_01(i32 %N) {
entry:
  %cmp27 = icmp sgt i32 %N, 1
  br i1 %cmp27, label %for.cond1.preheader.lr.ph, label %for.end16

for.cond1.preheader.lr.ph:
  %0 = add i32 %N, -1
  br label %for.body3.preheader

for.body3.preheader:
  %indvars.iv30 = phi i64 [ 1, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next31, %for.inc14 ]
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 1, %for.body3.preheader ]
  %1 = add nsw i64 %indvars.iv, -1
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %1, i64 %indvars.iv30
  %2 = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @B, i64 0, i64 %indvars.iv, i64 %indvars.iv30
  %3 = load i32, i32* %arrayidx9
  %add = add nsw i32 %3, %2
  %arrayidx13 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv30
  store i32 %add, i32* %arrayidx13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc14, label %for.body3

for.inc14:
  %indvars.iv.next31 = add nuw nsw i64 %indvars.iv30, 1
  %lftr.wideiv32 = trunc i64 %indvars.iv30 to i32
  %exitcond33 = icmp eq i32 %lftr.wideiv32, %0
  br i1 %exitcond33, label %for.end16, label %for.body3.preheader

for.end16:
  ret void
}

;; ---------------------------------------Test case 02---------------------------------
;; Check loop interchange profitability model. 
;; This tests profitability model when operands of getelementpointer and not exactly the induction variable but some 
;; arithmetic operation on them.
;;   for(int i=1;i<N;i++)
;;    for(int j=1;j<N;j++)
;;       A[j-1][i-1] = A[j - 1][i-1] + B[j-1][i-1];

; CHECK: Not interchanging loops. Cannot prove legality.

define void @interchange_02(i32 %N) {
entry:
  %cmp32 = icmp sgt i32 %N, 1
  br i1 %cmp32, label %for.cond1.preheader.lr.ph, label %for.end21

for.cond1.preheader.lr.ph:
  %0 = add i32 %N, -1
  br label %for.body3.lr.ph

for.body3.lr.ph:
  %indvars.iv35 = phi i64 [ 1, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next36, %for.inc19 ]
  %1 = add nsw i64 %indvars.iv35, -1
  br label %for.body3

for.body3: 
  %indvars.iv = phi i64 [ 1, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %2 = add nsw i64 %indvars.iv, -1
  %arrayidx6 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %2, i64 %1
  %3 = load i32, i32* %arrayidx6
  %arrayidx12 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @B, i64 0, i64 %2, i64 %1
  %4 = load i32, i32* %arrayidx12
  %add = add nsw i32 %4, %3
  store i32 %add, i32* %arrayidx6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc19, label %for.body3

for.inc19:
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %lftr.wideiv38 = trunc i64 %indvars.iv35 to i32
  %exitcond39 = icmp eq i32 %lftr.wideiv38, %0
  br i1 %exitcond39, label %for.end21, label %for.body3.lr.ph

for.end21:
  ret void
}

;;---------------------------------------Test case 03---------------------------------
;; Loops interchange is not profitable.
;;   for(int i=1;i<N;i++)
;;     for(int j=1;j<N;j++)
;;       A[i-1][j-1] = A[i - 1][j-1] + B[i][j];

; CHECK: Not interchanging loops. Cannot prove legality.

define void @interchange_03(i32 %N){
entry:
  %cmp31 = icmp sgt i32 %N, 1
  br i1 %cmp31, label %for.cond1.preheader.lr.ph, label %for.end19

for.cond1.preheader.lr.ph:
  %0 = add i32 %N, -1
  br label %for.body3.lr.ph

for.body3.lr.ph:
  %indvars.iv34 = phi i64 [ 1, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next35, %for.inc17 ]
  %1 = add nsw i64 %indvars.iv34, -1
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 1, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %2 = add nsw i64 %indvars.iv, -1
  %arrayidx6 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %1, i64 %2
  %3 = load i32, i32* %arrayidx6
  %arrayidx10 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @B, i64 0, i64 %indvars.iv34, i64 %indvars.iv
  %4 = load i32, i32* %arrayidx10
  %add = add nsw i32 %4, %3
  store i32 %add, i32* %arrayidx6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc17, label %for.body3

for.inc17:
  %indvars.iv.next35 = add nuw nsw i64 %indvars.iv34, 1
  %lftr.wideiv37 = trunc i64 %indvars.iv34 to i32
  %exitcond38 = icmp eq i32 %lftr.wideiv37, %0
  br i1 %exitcond38, label %for.end19, label %for.body3.lr.ph

for.end19:
  ret void
}
