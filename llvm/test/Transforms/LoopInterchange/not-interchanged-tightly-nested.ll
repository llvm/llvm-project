; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer
@C = common global [100 x [100 x i32]] zeroinitializer
@D = common global [100 x [100 x [100 x i32]]] zeroinitializer

;; Loops not tightly nested are not interchanged
;;  for(int j=0;j<N;j++) {
;;    B[j] = j+k;
;;    for(int i=0;i<N;i++)
;;      A[j][i] = A[j][i]+B[j];
;;  }

define void @interchange_05(i32 %k, i32 %N){
entry:
  %cmp30 = icmp sgt i32 %N, 0
  br i1 %cmp30, label %for.body.lr.ph, label %for.end17

for.body.lr.ph:
  %0 = add i32 %N, -1
  %1 = zext i32 %k to i64
  br label %for.body

for.body:
  %indvars.iv32 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next33, %for.inc15 ]
  %2 = add nsw i64 %indvars.iv32, %1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* @B, i64 0, i64 %indvars.iv32
  %3 = trunc i64 %2 to i32
  store i32 %3, i32* %arrayidx
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx7 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv32, i64 %indvars.iv
  %4 = load i32, i32* %arrayidx7
  %add10 = add nsw i32 %3, %4
  store i32 %add10, i32* %arrayidx7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc15, label %for.body3

for.inc15:
  %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
  %lftr.wideiv35 = trunc i64 %indvars.iv32 to i32
  %exitcond36 = icmp eq i32 %lftr.wideiv35, %0
  br i1 %exitcond36, label %for.end17, label %for.body

for.end17:
  ret void
}

; CHECK-LABEL: @interchange_05
; CHECK: entry:
; CHECK: %cmp30 = icmp sgt i32 %N, 0
; CHECK: br i1 %cmp30, label %for.body.lr.ph, label %for.end17
; CHECK: for.body.lr.ph:
; CHECK: %0 = add i32 %N, -1
; CHECK: %1 = zext i32 %k to i64
; CHECK: br label %for.body
; CHECK: for.body:
; CHECK: %indvars.iv32 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next33, %for.inc15 ]
; CHECK: %2 = add nsw i64 %indvars.iv32, %1
; CHECK: %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* @B, i64 0, i64 %indvars.iv32
; CHECK: %3 = trunc i64 %2 to i32
; CHECK: store i32 %3, i32* %arrayidx
; CHECK: br label %for.body3.preheader
; CHECK: for.body3.preheader:
; CHECK: br label %for.body3
; CHECK: for.body3:
; CHECK: %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 0, %for.body3.preheader ]
; CHECK: %arrayidx7 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv32, i64 %indvars.iv
; CHECK: %4 = load i32, i32* %arrayidx7
; CHECK: %add10 = add nsw i32 %3, %4
; CHECK: store i32 %add10, i32* %arrayidx7
; CHECK: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: %lftr.wideiv = trunc i64 %indvars.iv to i32
; CHECK: %exitcond = icmp eq i32 %lftr.wideiv, %0
; CHECK: br i1 %exitcond, label %for.inc15, label %for.body3
; CHECK: for.inc15:
; CHECK: %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
; CHECK: %lftr.wideiv35 = trunc i64 %indvars.iv32 to i32
; CHECK: %exitcond36 = icmp eq i32 %lftr.wideiv35, %0
; CHECK: br i1 %exitcond36, label %for.end17.loopexit, label %for.body
; CHECK: for.end17.loopexit:
; CHECK: br label %for.end17
; CHECK: for.end17:
; CHECK: ret void


declare void @foo(...) readnone

;; Loops not tightly nested are not interchanged
;;  for(int j=0;j<N;j++) {
;;    foo();
;;    for(int i=2;i<N;i++)
;;      A[j][i] = A[j][i]+k;
;;  }

define void @interchange_06(i32 %k, i32 %N) {
entry:
  %cmp22 = icmp sgt i32 %N, 0
  br i1 %cmp22, label %for.body.lr.ph, label %for.end12

for.body.lr.ph:
  %0 = add i32 %N, -1
  br label %for.body

for.body:
  %indvars.iv24 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next25, %for.inc10 ]
  tail call void (...) @foo()
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 2, %for.body ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv24, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx5
  %add = add nsw i32 %1, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:
  %indvars.iv.next25 = add nuw nsw i64 %indvars.iv24, 1
  %lftr.wideiv26 = trunc i64 %indvars.iv24 to i32
  %exitcond27 = icmp eq i32 %lftr.wideiv26, %0
  br i1 %exitcond27, label %for.end12, label %for.body

for.end12:
  ret void
}
;; Here we are checking if the inner phi is not split then we have not interchanged.
; CHECK-LABEL: @interchange_06
; CHECK:  phi i64 [ %indvars.iv.next, %for.body3 ], [ 2, %for.body3.preheader ]
; CHECK-NEXT: getelementptr
; CHECK-NEXT: %1 = load
