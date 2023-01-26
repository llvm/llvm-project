; REQUIRES: asserts
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info -stats 2>&1 | FileCheck -check-prefix=STATS %s
; RUN: FileCheck --input-file=%t %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = common global [100 x [100 x i32]] zeroinitializer

declare void @foo(i64 %a)
declare void @bar(i64 %a) readnone

;;--------------------------------------Test case 01------------------------------------
;; Not safe to interchange, because the called function `foo` is not marked as
;; readnone, so it could introduce dependences.
;;
;;  for(int i=0;i<100;i++) {
;;    for(int j=1;j<100;j++) {
;;      foo(i);
;;      A[j][i] = A[j][i]+k;
;;    }
;; }

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            CallInst
; CHECK-NEXT: Function:        interchange_01
; CHECK-NEXT: Args:
; CHECK-NEXT: - String:          Cannot interchange loops due to call instruction.

define void @interchange_01(i32 %k) {
entry:
  br label %for1.header

for1.header:
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for1.inc10 ]
  br label %for2

for2:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for2 ], [ 1, %for1.header ]
  call void @foo(i64 %indvars.iv23)
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %lv = load i32, ptr %arrayidx5
  %add = add nsw i32 %lv, %k
  store i32 %add, ptr %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 99
  br i1 %exitcond, label %for2.loopexit , label %for2

for2.loopexit:
  br label %for1.inc10

for1.inc10:
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond26 = icmp eq i64 %indvars.iv23, 99
  br i1 %exitcond26, label %for1.loopexit, label %for1.header

for1.loopexit:
  br label %exit

exit:
  ret void
}

;;--------------------------------------Test case 02------------------------------------
;; Safe to interchange, because the called function `bar` is marked as readnone,
;; so it cannot introduce dependences.
;;
;;  for(int i=0;i<100;i++) {
;;    for(int j=1;j<100;j++) {
;;      bar(i);
;;      A[j][i] = A[j][i]+k;
;;    }
;; }

; CHECK: --- !Passed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        interchange_02
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT: ...

define void @interchange_02(i32 %k) {
entry:
  br label %for1.header

for1.header:
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for1.inc10 ]
  br label %for2

for2:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for2 ], [ 1, %for1.header ]
  call void @bar(i64 %indvars.iv23)
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %lv = load i32, ptr %arrayidx5
  %add = add nsw i32 %lv, %k
  store i32 %add, ptr %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 99
  br i1 %exitcond, label %for2.loopexit , label %for2

for2.loopexit:
  br label %for1.inc10

for1.inc10:
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond26 = icmp eq i64 %indvars.iv23, 99
  br i1 %exitcond26, label %for1.loopexit, label %for1.header

for1.loopexit:
  br label %exit

exit:
  ret void
}

; Check stats, we interchanged 1 out of 2 loops.
; STATS: 1 loop-interchange - Number of loops interchanged
