; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
;
; Check that we will recognize this SCoP.
;
;    void f(int *A, long N) {
;      int j = 0;
;      while (N > j) {
;        int x = A[0];
;        int i = 1;
;        do {
;          A[x] = 42;
;          A += x;
;        } while (i++ < N);
;      }
;    }
;
; CHECK: Valid Region for Scop: bb1 => bb0
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i64 %N) {
bb:
  br label %bb0

bb0:
  %j = phi i64 [ %j.next, %bb1 ], [ 1, %bb ]
  %tmp = load i32, ptr %A, align 4
  %exitcond0 = icmp sgt i64 %N, %j
  %j.next = add nuw nsw i64 %j, 1
  br i1 %exitcond0, label %bb1, label %bb13

bb1:                                              ; preds = %bb7, %bb0
  %i = phi i64 [ %i.next, %bb1 ], [ 1, %bb0 ]
  %.0 = phi ptr [ %A, %bb0 ], [ %tmp12, %bb1 ]
  %tmp8 = sext i32 %tmp to i64
  %tmp9 = getelementptr inbounds i32, ptr %.0, i64 %tmp8
  store i32 42, ptr %tmp9, align 4
  %tmp11 = sext i32 %tmp to i64
  %tmp12 = getelementptr inbounds i32, ptr %.0, i64 %tmp11
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i, %N
  br i1 %exitcond, label %bb1, label %bb0

bb13:                                             ; preds = %bb1
  ret void
}
