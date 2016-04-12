; RUN: opt %loadPolly -polly-opt-isl -polly-codegen -polly-vectorizer=polly -polly-prevect-width=8 -S -dce < %s | FileCheck %s
;
;    void foo(long n, float A[restrict][n], float B[restrict][n],
;             float C[restrict][n], float D[restrict][n]) {
;      for (long i = 0; i < 8; i++)
;        for (long j = 0; j < 8; j++)
;          A[i][j] += B[i][0] + C[i][2 * j] + D[j][0];
;    }
;

; CHECK: shufflevector
; CHECK: insertelement
; CHECK: insertelement
; CHECK: insertelement
; CHECK: insertelement
; CHECK: insertelement
; CHECK: insertelement
; CHECK: insertelement
; CHECK: insertelement
; CHECK: store <8 x float>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %n, float* noalias %A, float* noalias %B, float* noalias %C, float* noalias %D) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb25, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp26, %bb25 ]
  %exitcond2 = icmp ne i64 %i.0, 8
  br i1 %exitcond2, label %bb4, label %bb27

bb4:                                              ; preds = %bb3
  br label %bb5

bb5:                                              ; preds = %bb22, %bb4
  %j.0 = phi i64 [ 0, %bb4 ], [ %tmp23, %bb22 ]
  %exitcond = icmp ne i64 %j.0, 8
  br i1 %exitcond, label %bb6, label %bb24

bb6:                                              ; preds = %bb5
  %tmp = mul nsw i64 %i.0, %n
  %tmp7 = getelementptr inbounds float, float* %B, i64 %tmp
  %tmp8 = load float, float* %tmp7, align 4
  %tmp9 = shl nsw i64 %j.0, 1
  %tmp10 = mul nsw i64 %i.0, %n
  %.sum = add i64 %tmp10, %tmp9
  %tmp11 = getelementptr inbounds float, float* %C, i64 %.sum
  %tmp12 = load float, float* %tmp11, align 4
  %tmp13 = fadd float %tmp8, %tmp12
  %tmp14 = mul nsw i64 %j.0, %n
  %tmp15 = getelementptr inbounds float, float* %D, i64 %tmp14
  %tmp16 = load float, float* %tmp15, align 4
  %tmp17 = fadd float %tmp13, %tmp16
  %tmp18 = mul nsw i64 %i.0, %n
  %.sum1 = add i64 %tmp18, %j.0
  %tmp19 = getelementptr inbounds float, float* %A, i64 %.sum1
  %tmp20 = load float, float* %tmp19, align 4
  %tmp21 = fadd float %tmp20, %tmp17
  store float %tmp21, float* %tmp19, align 4
  br label %bb22

bb22:                                             ; preds = %bb6
  %tmp23 = add nsw i64 %j.0, 1
  br label %bb5

bb24:                                             ; preds = %bb5
  br label %bb25

bb25:                                             ; preds = %bb24
  %tmp26 = add nsw i64 %i.0, 1
  br label %bb3

bb27:                                             ; preds = %bb3
  ret void
}
