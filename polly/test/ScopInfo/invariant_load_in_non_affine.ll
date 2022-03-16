; RUN: opt %loadPolly -polly-print-detect -disable-output \
; RUN:   -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
; CHECK-NOT: Valid Region for Scop
;
;    void foo(float A[], float B[], long *p) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          if (B[i])
;            A[i * (*p) + j] += i * j;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, float* %B, i64* %p) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb21, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp22, %bb21 ]
  %exitcond1 = icmp ne i64 %i.0, 1024
  br i1 %exitcond1, label %bb3, label %bb23

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb18, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp19, %bb18 ]
  %exitcond = icmp ne i64 %j.0, 1024
  br i1 %exitcond, label %bb5, label %bb20

bb5:                                              ; preds = %bb4
  %tmp = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp6 = load float, float* %tmp, align 4
  %tmp7 = fcmp une float %tmp6, 0.000000e+00
  br i1 %tmp7, label %bb8, label %bb17

bb8:                                              ; preds = %bb5
  %tmp9 = mul nuw nsw i64 %i.0, %j.0
  %tmp10 = sitofp i64 %tmp9 to float
  %tmp11 = load i64, i64* %p, align 8
  %tmp12 = mul nsw i64 %i.0, %tmp11
  %tmp13 = add nsw i64 %tmp12, %j.0
  %tmp14 = getelementptr inbounds float, float* %A, i64 %tmp13
  %tmp15 = load float, float* %tmp14, align 4
  %tmp16 = fadd float %tmp15, %tmp10
  store float %tmp16, float* %tmp14, align 4
  br label %bb17

bb17:                                             ; preds = %bb8, %bb5
  br label %bb18

bb18:                                             ; preds = %bb17
  %tmp19 = add nuw nsw i64 %j.0, 1
  br label %bb4

bb20:                                             ; preds = %bb4
  br label %bb21

bb21:                                             ; preds = %bb20
  %tmp22 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb23:                                             ; preds = %bb2
  ret void
}
