; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=IR %s

; REQUIRES: pollyacc

;    void foo(long n, float A[][n]) {
;      for (long i = 0; i < 32; i++)
;        for (long j = 0; j < 32; j++)
;          A[i][j] += A[i + 1][j + 1];
;    }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; This test case failed at some point as %n was only available in this kernel
; when referenced through an isl_id in an isl ast expression, but not when
; it was referenced from a SCEV  or instruction that not part of any loop
; bound.

; IR: %polly.access.mul.MemRef_A6 = mul nsw i64 {{.*}}, %n

define void @foo(i64 %n, float* %A) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb19, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp20, %bb19 ]
  %exitcond1 = icmp ne i64 %i.0, 32
  br i1 %exitcond1, label %bb3, label %bb21

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb16, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp17, %bb16 ]
  %exitcond = icmp ne i64 %j.0, 32
  br i1 %exitcond, label %bb5, label %bb18

bb5:                                              ; preds = %bb4
  %tmp = add nuw nsw i64 %j.0, 1
  %tmp6 = add nuw nsw i64 %i.0, 1
  %tmp7 = mul nsw i64 %tmp6, %n
  %tmp8 = getelementptr inbounds float, float* %A, i64 %tmp7
  %tmp9 = getelementptr inbounds float, float* %tmp8, i64 %tmp
  %tmp10 = load float, float* %tmp9, align 4
  %tmp11 = mul nsw i64 %i.0, %n
  %tmp12 = getelementptr inbounds float, float* %A, i64 %tmp11
  %tmp13 = getelementptr inbounds float, float* %tmp12, i64 %j.0
  %tmp14 = load float, float* %tmp13, align 4
  %tmp15 = fadd float %tmp14, %tmp10
  store float %tmp15, float* %tmp13, align 4
  br label %bb16

bb16:                                             ; preds = %bb5
  %tmp17 = add nuw nsw i64 %j.0, 1
  br label %bb4

bb18:                                             ; preds = %bb4
  br label %bb19

bb19:                                             ; preds = %bb18
  %tmp20 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb21:                                             ; preds = %bb2
  ret void
}
