; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer

declare void @foo(i64 %a)
declare void @bar(i64 %a) readnone

;;--------------------------------------Test case 01------------------------------------
;; Not safe to interchange, because the called function `foo` is not marked as
;; readnone, so it could introduce dependences.
;;
;;  for(int i=0;i<N;i++) {
;;    for(int j=1;j<N;j++) {
;;      foo(i);
;;      A[j][i] = A[j][i]+k;
;;    }
;; }

define void @interchange_01(i32 %k, i32 %N) {
entry:
  %cmp21 = icmp sgt i32 %N, 0
  br i1 %cmp21, label %for1.ph, label %exit

for1.ph:
  %cmp219 = icmp sgt i32 %N, 1
  %0 = add i32 %N, -1
  br label %for1.header

for1.header:
  %indvars.iv23 = phi i64 [ 0, %for1.ph ], [ %indvars.iv.next24, %for1.inc10 ]
  br i1 %cmp219, label %for2.ph, label %for1.inc10

for2.ph:
  br label %for2

for2:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for2 ], [ 1, %for2.ph ]
  call void @foo(i64 %indvars.iv23)
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %1 = load i32, i32* %arrayidx5
  %add = add nsw i32 %1, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for2.loopexit , label %for2

for2.loopexit:
  br label %for1.inc10

for1.inc10:
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv23 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, %0
  br i1 %exitcond26, label %for1.loopexit, label %for1.header

for1.loopexit:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @interchange_01
; CHECK: for1.ph:
; CHECK: br label %for1.header

; CHECK: for1.header:
; CHECK-NEXT: %indvars.iv23 = phi i64 [ 0, %for1.ph ], [ %indvars.iv.next24, %for1.inc10 ]
; CHECK-NEXT: br i1 %cmp219, label %for2.ph, label %for1.inc10

; CHECK: for2:
; CHECK: br i1 %exitcond, label %for2.loopexit, label %for2

; CHECK: for1.inc10:
; CHECK: br i1 %exitcond26, label %for1.loopexit, label %for1.header

; CHECK: for1.loopexit:
; CHECK-NEXT: br label %exit


;;--------------------------------------Test case 02------------------------------------
;; Safe to interchange, because the called function `bar` is marked as readnone,
;; so it cannot introduce dependences.
;;
;;  for(int i=0;i<N;i++) {
;;    for(int j=1;j<N;j++) {
;;      bar(i);
;;      A[j][i] = A[j][i]+k;
;;    }
;; }

define void @interchange_02(i32 %k, i32 %N) {
entry:
  %cmp21 = icmp sgt i32 %N, 0
  br i1 %cmp21, label %for1.ph, label %exit

for1.ph:
  %cmp219 = icmp sgt i32 %N, 1
  %0 = add i32 %N, -1
  br label %for1.header

for1.header:
  %indvars.iv23 = phi i64 [ 0, %for1.ph ], [ %indvars.iv.next24, %for1.inc10 ]
  br i1 %cmp219, label %for2.ph, label %for1.inc10

for2.ph:
  br label %for2

for2:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for2 ], [ 1, %for2.ph ]
  call void @bar(i64 %indvars.iv23)
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %1 = load i32, i32* %arrayidx5
  %add = add nsw i32 %1, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for2.loopexit , label %for2

for2.loopexit:
  br label %for1.inc10

for1.inc10:
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv23 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, %0
  br i1 %exitcond26, label %for1.loopexit, label %for1.header

for1.loopexit:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @interchange_02
; CHECK: for1.header:
; CHECK-NEXT: %indvars.iv23 = phi i64 [ 0, %for1.ph ], [ %indvars.iv.next24, %for1.inc10 ]
; CHECK-NEXT: br i1 %cmp219, label %for2.split1, label %for1.loopexit

; CHECK: for2.split1:
; CHECK: br label %for2.loopexit

; CHECK: for2.split:
; CHECK-NEXT: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: br i1 %exitcond, label %for1.loopexit, label %for2

; CHECK: for2.loopexit:
; CHECK-NEXT:  br label %for1.inc10

; CHECK: for1.inc10:
; CHECK: br i1 %exitcond26, label %for2.split, label %for1.header
