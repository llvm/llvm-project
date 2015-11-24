; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; void foo(float *A, float *B, float *C, long N) {
; 	for (long i = 0; i < N; i++)
; 		C[i] = A[i] + B[i];
; }
;
; CHECK: Alias Groups (2):
;
; This test case verifies that we do not create run-time checks for two
; read-only arrays.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, float* %B, float* %C, i64 %N) {
entry:
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx.A = getelementptr float, float* %A, i64 %indvar
  %arrayidx.B = getelementptr float, float* %B, i64 %indvar
  %arrayidx.C = getelementptr float, float* %C, i64 %indvar
  %val.A = load float, float* %arrayidx.A
  %val.B = load float, float* %arrayidx.B
  %add = fadd float %val.A, %val.B
  store float %add, float* %arrayidx.C
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, %N
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}
