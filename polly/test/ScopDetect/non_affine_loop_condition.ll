; RUN: opt %loadPolly -polly-allow-nonaffine-loops                                   -polly-print-detect -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-allow-nonaffine-loops -polly-process-unprofitable=false -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=PROFIT
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++) {
;        while (A[i])
;          A[i]--;
;      }
;    }
;
; PROFIT-NOT: Valid
;
; CHECK: Valid Region for Scop: bb1 => bb12
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb11, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb11 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb12

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb6, %bb2
  %tmp = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp4 = load i32, i32* %tmp, align 4
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb10, label %bb6

bb6:                                              ; preds = %bb3
  %tmp7 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp8 = load i32, i32* %tmp7, align 4
  %tmp9 = add nsw i32 %tmp8, -1
  store i32 %tmp9, i32* %tmp7, align 4
  br label %bb3

bb10:                                             ; preds = %bb3
  br label %bb11

bb11:                                             ; preds = %bb10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb12:                                             ; preds = %bb1
  ret void
}
