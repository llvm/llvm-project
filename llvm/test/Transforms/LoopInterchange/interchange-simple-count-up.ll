; REQUIRES: asserts
; RUN: opt < %s -basicaa -loop-interchange -verify-dom-info -S -debug 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer

;;  for(int i=0;i<N;i++)
;;    for(int j=1;j<N;j++)
;;      A[j][i] = A[j][i]+k;

; CHECK: Not interchanging loops. Cannot prove legality.

define void @interchange_01(i32 %k, i32 %N) {
entry:
  %cmp21 = icmp sgt i32 %N, 0
  br i1 %cmp21, label %for.cond1.preheader.lr.ph, label %for.end12

for.cond1.preheader.lr.ph:
  %cmp219 = icmp sgt i32 %N, 1
  %0 = add i32 %N, -1
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv23 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next24, %for.inc10 ]
  br i1 %cmp219, label %for.body3, label %for.inc10

for.body3:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 1, %for.cond1.preheader ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %1 = load i32, i32* %arrayidx5
  %add = add nsw i32 %1, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv23 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, %0
  br i1 %exitcond26, label %for.end12, label %for.cond1.preheader

for.end12:
  ret void
}
