; RUN: opt %loadPolly -polly-process-unprofitable=false \
; RUN: \
; RUN: -polly-detect -analyze < %s | FileCheck %s

; RUN: opt %loadPolly -polly-process-unprofitable=true \
; RUN: \
; RUN: -polly-detect -analyze < %s | FileCheck %s

; CHECK: Valid Region for Scop:

;    void foo(float *A, float *B, long N) {
;      if (N > 100)
;        for (long i = 0; i < 100; i++)
;          A[i] += i;
;      else
;        for (long i = 0; i < 100; i++)
;          B[i] += i;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, float* %B, i64 %N) {
entry:
  br label %bb

bb:
  %tmp = icmp sgt i64 %N, 100
  br i1 %tmp, label %bb2, label %bb12

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb9, %bb2
  %i.0 = phi i64 [ 0, %bb2 ], [ %tmp10, %bb9 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb4, label %bb11

bb4:                                              ; preds = %bb3
  %tmp5 = sitofp i64 %i.0 to float
  %tmp6 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp7 = load float, float* %tmp6, align 4
  %tmp8 = fadd float %tmp7, %tmp5
  store float %tmp8, float* %tmp6, align 4
  br label %bb9

bb9:                                              ; preds = %bb4
  %tmp10 = add nsw i64 %i.0, 1
  br label %bb3

bb11:                                             ; preds = %bb3
  br label %bb22

bb12:                                             ; preds = %bb
  br label %bb13

bb13:                                             ; preds = %bb19, %bb12
  %i1.0 = phi i64 [ 0, %bb12 ], [ %tmp20, %bb19 ]
  %exitcond1 = icmp ne i64 %i1.0, 100
  br i1 %exitcond1, label %bb14, label %bb21

bb14:                                             ; preds = %bb13
  %tmp15 = sitofp i64 %i1.0 to float
  %tmp16 = getelementptr inbounds float, float* %B, i64 %i1.0
  %tmp17 = load float, float* %tmp16, align 4
  %tmp18 = fadd float %tmp17, %tmp15
  store float %tmp18, float* %tmp16, align 4
  br label %bb19

bb19:                                             ; preds = %bb14
  %tmp20 = add nsw i64 %i1.0, 1
  br label %bb13

bb21:                                             ; preds = %bb13
  br label %bb22

bb22:                                             ; preds = %bb21, %bb11
  ret void
}
