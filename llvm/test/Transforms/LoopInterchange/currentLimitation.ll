; RUN: opt < %s -basic-aa -loop-interchange -pass-remarks-missed='loop-interchange' \
; RUN:   -pass-remarks-output=%t -verify-loop-info -verify-dom-info -S | FileCheck -check-prefix=IR %s
; RUN: FileCheck --input-file=%t %s

; RUN: opt < %s -basic-aa -loop-interchange -pass-remarks-missed='loop-interchange' \
; RUN:   -da-disable-delinearization-checks -pass-remarks-output=%t             \
; RUN:   -verify-loop-info -verify-dom-info -S | FileCheck -check-prefix=IR %s
; RUN: FileCheck --check-prefix=DELIN --input-file=%t %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
 
@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x [100 x [100 x i32]]] zeroinitializer
@C = common global [100 x [100 x i64]] zeroinitializer
 
;;--------------------------------------Test case 01------------------------------------
;; [FIXME] This loop though valid is currently not interchanged due to the limitation that we cannot split the inner loop latch due to multiple use of inner induction
;; variable.(used to increment the loop counter and to access A[j+1][i+1]
;;  for(int i=0;i<N-1;i++)
;;    for(int j=1;j<N-1;j++)
;;      A[j+1][i+1] = A[j+1][i+1] + k;

; IR-LABEL: @interchange_01
; IR-NOT: split

; CHECK:      Name:            Dependence
; CHECK-NEXT: Function:        interchange_01

; DELIN:      Name:            UnsupportedInsBetweenInduction
; DELIN-NEXT: Function:        interchange_01
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

; When currently cannot interchange this loop, because transform currently
; expects the latches to be the exiting blocks too.

; IR-LABEL: @interchange_02
; IR-NOT: split
;
; CHECK:      Name:            ExitingNotLatch
; CHECK-NEXT: Function:        interchange_02
define void @interchange_02(i64 %k, i64 %N) {
entry:
  br label %for1.header

for1.header:
  %j23 = phi i64 [ 0, %entry ], [ %j.next24, %for1.inc10 ]
  br label %for2

for2:
  %j = phi i64 [ %j.next, %latch ], [ 0, %for1.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* @C, i64 0, i64 %j, i64 %j23
  %lv = load i64, i64* %arrayidx5
  %add = add nsw i64 %lv, %k
  store i64 %add, i64* %arrayidx5
  %exitcond = icmp eq i64 %j, 99
  br i1 %exitcond, label %for1.inc10, label %latch
latch:
  %j.next = add nuw nsw i64 %j, 1
  br label %for2

for1.inc10:
  %j.next24 = add nuw nsw i64 %j23, 1
  %exitcond26 = icmp eq i64 %j23, 99
  br i1 %exitcond26, label %for.end12, label %for1.header

for.end12:
  ret void
}
