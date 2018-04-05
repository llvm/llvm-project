; REQUIRES: asserts
; RUN: opt < %s -basicaa -loop-interchange -verify-dom-info -verify-loop-info \
; RUN:     -S -debug 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
 
@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x [100 x [100 x i32]]] zeroinitializer
 
;;--------------------------------------Test case 01------------------------------------
;; [FIXME] This loop though valid is currently not interchanged due to the limitation that we cannot split the inner loop latch due to multiple use of inner induction
;; variable.(used to increment the loop counter and to access A[j+1][i+1]
;;  for(int i=0;i<N-1;i++)
;;    for(int j=1;j<N-1;j++)
;;      A[j+1][i+1] = A[j+1][i+1] + k;

; CHECK: Not interchanging loops. Cannot prove legality.

define void @interchange_01(i32 %k, i32 %N) {
 entry:
   %sub = add nsw i32 %N, -1
   %cmp26 = icmp sgt i32 %N, 1
   br i1 %cmp26, label %for.cond1.preheader.lr.ph, label %for.end17
 
 for.cond1.preheader.lr.ph:
   %cmp324 = icmp sgt i32 %sub, 1
   %0 = add i32 %N, -2
   %1 = sext i32 %sub to i64
   br label %for.cond1.preheader
 
 for.cond.loopexit:
   %cmp = icmp slt i64 %indvars.iv.next29, %1
   br i1 %cmp, label %for.cond1.preheader, label %for.end17
 
 for.cond1.preheader:
   %indvars.iv28 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next29, %for.cond.loopexit ]
   %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
   br i1 %cmp324, label %for.body4, label %for.cond.loopexit
 
 for.body4:
   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body4 ], [ 1, %for.cond1.preheader ]
   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
   %arrayidx7 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv.next, i64 %indvars.iv.next29
   %2 = load i32, i32* %arrayidx7
   %add8 = add nsw i32 %2, %k
   store i32 %add8, i32* %arrayidx7
   %lftr.wideiv = trunc i64 %indvars.iv to i32
   %exitcond = icmp eq i32 %lftr.wideiv, %0
   br i1 %exitcond, label %for.cond.loopexit, label %for.body4
 
 for.end17: 
   ret void
}
