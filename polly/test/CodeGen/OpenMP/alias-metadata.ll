; RUN: opt -opaque-pointers=0 %loadPolly -polly-codegen -polly-parallel -S < %s | FileCheck %s
;
;    void foo(float *A, float *B) {
;      for (long i = 0; i < 1000; i++)
;        for (long j = 0; j < 1000; j++)
;          A[i] = B[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: define internal void @foo_polly_subfn

define void @foo(float* %A, float* %B) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb11, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  %exitcond1 = icmp ne i64 %i.0, 1000
  br i1 %exitcond1, label %bb3, label %bb13

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb8, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp9, %bb8 ]
  %exitcond = icmp ne i64 %j.0, 1000
  br i1 %exitcond, label %bb5, label %bb10

bb5:                                              ; preds = %bb4
  %tmp = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp7 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp6 = load float, float* %tmp, align 4
  store float %tmp6, float* %tmp7, align 4
; CHECK: %tmp6_p_scalar_ = load float, float* %scevgep, align 4, !alias.scope !0, !noalias !3
; CHECK: store float %tmp6_p_scalar_, float* %scevgep8, align 4, !alias.scope !3, !noalias !0
  br label %bb8

bb8:                                              ; preds = %bb5
  %tmp9 = add nsw i64 %j.0, 1
  br label %bb4

bb10:                                             ; preds = %bb4
  br label %bb11

bb11:                                             ; preds = %bb10
  %tmp12 = add nsw i64 %i.0, 1
  br label %bb2

bb13:                                             ; preds = %bb2
  ret void
}
