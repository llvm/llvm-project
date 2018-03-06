; RUN: opt < %s -basicaa -loop-interchange -verify-dom-info -S -pass-remarks=loop-interchange 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer

;; for(int i=0;i<100;i++)
;;   for(int j=100;j>=0;j--)
;;     A[j][i] = A[j][i]+k;

; CHECK: Loop interchanged with enclosing loop.

define void @interchange_02(i32 %k) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv19 = phi i64 [ 0, %entry ], [ %indvars.iv.next20, %for.inc10 ]
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 100, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv19
  %0 = load i32, i32* %arrayidx5
  %add = add nsw i32 %0, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %cmp2 = icmp sgt i64 %indvars.iv, 0
  br i1 %cmp2, label %for.body3, label %for.inc10

for.inc10:
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %exitcond = icmp eq i64 %indvars.iv.next20, 100
  br i1 %exitcond, label %for.end11, label %for.cond1.preheader

for.end11:
  ret void
}
