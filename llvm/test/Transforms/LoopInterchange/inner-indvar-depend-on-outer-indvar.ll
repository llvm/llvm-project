; REQUIRES: asserts
; RUN: opt < %s -basic-aa -loop-interchange -verify-dom-info -verify-loop-info \
; RUN:     -S -debug 2>&1 | FileCheck %s

target triple = "powerpc64le-unknown-linux-gnu"
@A = common global [100 x [100 x i64]] zeroinitializer
@N = dso_local local_unnamed_addr global i64 100, align 8


;;  for(int i=0;i<100;i++)
;;    for(int j=0;j<i;j++)
;;      A[j][i] = A[j][i]+k;

;; Inner loop induction variable exit condition depends on the
;; outer loop induction variable, i.e., triangular loops.
; CHECK: Loop structure not understood by pass
; CHECK: Not interchanging loops. Cannot prove legality.

define void @interchange_01(i64 %k) {
entry:
  br label %for1.header

for1.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for1.inc10 ]
  br label %for2

for2:
  %j = phi i64 [ %j.next, %for2 ], [ 0, %for1.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* @A, i64 0, i64 %j, i64 %i
  %lv = load i64, i64* %arrayidx5
  %add = add nsw i64 %lv, %k
  store i64 %add, i64* %arrayidx5
  %j.next = add nuw nsw i64 %j, 1
  %exitcond = icmp eq i64 %j, %i
  br i1 %exitcond, label %for1.inc10, label %for2

for1.inc10:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond26 = icmp eq i64 %i, 99
  br i1 %exitcond26, label %for.end12, label %for1.header

for.end12:
  ret void
}


;;  for(int i=0;i<100;i++)
;;    for(int j=0;j+i<100;j++)
;;      A[j][i] = A[j][i]+k;

;; Inner loop induction variable exit condition depends on the
;; outer loop induction variable, i.e., triangular loops.
; CHECK: Loop structure not understood by pass
; CHECK: Not interchanging loops. Cannot prove legality.

define void @interchange_02(i64 %k) {
entry:
  br label %for1.header

for1.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for1.inc10 ]
  br label %for2

for2:
  %j = phi i64 [ %j.next, %for2 ], [ 0, %for1.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* @A, i64 0, i64 %j, i64 %i
  %lv = load i64, i64* %arrayidx5
  %add = add nsw i64 %lv, %k
  store i64 %add, i64* %arrayidx5
  %0 = add nuw nsw i64 %j, %i
  %j.next = add nuw nsw i64 %j, 1
  %exitcond = icmp eq i64 %0, 100
  br i1 %exitcond, label %for1.inc10, label %for2

for1.inc10:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond26 = icmp eq i64 %i, 99
  br i1 %exitcond26, label %for.end12, label %for1.header

for.end12:
  ret void
}

;;  for(int i=0;i<100;i++)
;;    for(int j=0;i>j;j++)
;;      A[j][i] = A[j][i]+k;

;; Inner loop induction variable exit condition depends on the
;; outer loop induction variable, i.e., triangular loops.
; CHECK: Loop structure not understood by pass
; CHECK: Not interchanging loops. Cannot prove legality.

define void @interchange_03(i64 %k) {
entry:
  br label %for1.header

for1.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for1.inc10 ]
  br label %for2

for2:
  %j = phi i64 [ %j.next, %for2 ], [ 0, %for1.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* @A, i64 0, i64 %j, i64 %i
  %lv = load i64, i64* %arrayidx5
  %add = add nsw i64 %lv, %k
  store i64 %add, i64* %arrayidx5
  %j.next = add nuw nsw i64 %j, 1
  %exitcond = icmp ne i64 %i, %j
  br i1 %exitcond, label %for2, label %for1.inc10

for1.inc10:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond26 = icmp eq i64 %i, 99
  br i1 %exitcond26, label %for.end12, label %for1.header

for.end12:
  ret void
}

;;  for(int i=0;i<100;i++)
;;    for(int j=0;N>j;j++)
;;      A[j][i] = A[j][i]+k;

;; Inner loop induction variable exit condition depends on
;; an outer loop invariant, can do interchange.
; CHECK: Loops interchanged

define void @interchange_04(i64 %k) {
entry:
  %0 = load i64, i64* @N, align 4
  br label %for1.header

for1.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for1.inc10 ]
  br label %for2

for2:
  %j = phi i64 [ %j.next, %for2 ], [ 0, %for1.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* @A, i64 0, i64 %j, i64 %i
  %lv = load i64, i64* %arrayidx5
  %add = add nsw i64 %lv, %k
  store i64 %add, i64* %arrayidx5
  %j.next = add nuw nsw i64 %j, 1
  %exitcond = icmp ne i64 %0, %j
  br i1 %exitcond, label %for2, label %for1.inc10

for1.inc10:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond26 = icmp eq i64 %i, 99
  br i1 %exitcond26, label %for.end12, label %for1.header

for.end12:
  ret void
}
