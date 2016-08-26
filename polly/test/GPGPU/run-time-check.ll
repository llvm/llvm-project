; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR
;
; REQUIRES: pollyacc
;
;    void foo(long n, float A[][32]) {
;      for (long i = 0; i < n; i++)
;        for (long j = 0; j < n; j++)
;          A[i][j] += A[i + 1][j + 1];
;    }

; IR:       %tmp = icmp slt i64 %i.0, %n
; IR-NEXT:  br i1 %tmp, label %bb2, label %polly.merge_new_and_old

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %n, [32 x float]* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb15, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp16, %bb15 ]
  %tmp = icmp slt i64 %i.0, %n
  br i1 %tmp, label %bb2, label %bb17

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb12, %bb2
  %j.0 = phi i64 [ 0, %bb2 ], [ %tmp13, %bb12 ]
  %exitcond = icmp ne i64 %j.0, %n
  br i1 %exitcond, label %bb4, label %bb14

bb4:                                              ; preds = %bb3
  %tmp5 = add nuw nsw i64 %j.0, 1
  %tmp6 = add nuw nsw i64 %i.0, 1
  %tmp7 = getelementptr inbounds [32 x float], [32 x float]* %A, i64 %tmp6, i64 %tmp5
  %tmp8 = load float, float* %tmp7, align 4
  %tmp9 = getelementptr inbounds [32 x float], [32 x float]* %A, i64 %i.0, i64 %j.0
  %tmp10 = load float, float* %tmp9, align 4
  %tmp11 = fadd float %tmp10, %tmp8
  store float %tmp11, float* %tmp9, align 4
  br label %bb12

bb12:                                             ; preds = %bb4
  %tmp13 = add nuw nsw i64 %j.0, 1
  br label %bb3

bb14:                                             ; preds = %bb3
  br label %bb15

bb15:                                             ; preds = %bb14
  %tmp16 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb17:                                             ; preds = %bb1
  ret void
}
