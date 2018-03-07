; REQUIRES: asserts
; RUN: opt < %s -basicaa -loop-interchange -S -debug 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer

;; Loops should not be interchanged in this case as it is not profitable.
;;  for(int i=0;i<100;i++)
;;    for(int j=0;j<100;j++)
;;      A[i][j] = A[i][j]+k;

; CHECK: Interchanging loops not profitable.

define void @interchange_03(i32 %k) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc10 ]
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv21, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx5
  %add = add nsw i32 %0, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 100
  br i1 %exitcond23, label %for.end12, label %for.cond1.preheader

for.end12:
  ret void
}
