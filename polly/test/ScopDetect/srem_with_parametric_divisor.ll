; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
;
; CHECK-NOT: Valid Region for Scop:
;
;    void foo(float *A, long n, long p) {
;      for (long i = 0; i < 100; i++)
;        A[n % p] += 1;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, i64 %n, i64 %p) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = srem i64 %n, %p
  %tmp3 = getelementptr inbounds float, float* %A, i64 %tmp
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, 1.000000e+00
  store float %tmp5, float* %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
