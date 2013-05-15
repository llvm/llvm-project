; RUN: opt %loadPolly -basicaa -polly-scops -analyze -polly-allow-nonaffine < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo(long n, double A[], int INDEX[]) {
;   for (long i = 0; i < n; i++)
;     A[INDEX[i]] = i;
; }

define void @foo(i64 %n, double* noalias %A, i64* noalias %INDEX) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64* %INDEX, i64 %i
  %val = load i64* %arrayidx
  %arrayidx1 = getelementptr inbounds double* %A, i64 %val
  store double 1.0, double* %arrayidx1
  %inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK: p0: %n

; CHECK: Domain
; CHECK:   [n] -> { Stmt_for_body[i0] : i0 >= 0 and i0 <= -1 + n };
; CHECK: Scattering
; CHECK:   [n] -> { Stmt_for_body[i0] -> scattering[0, i0, 0] };
; CHECK: ReadAccess
; CHECK:   [n] -> { Stmt_for_body[i0] -> MemRef_INDEX[i0] };
; CHECK: WriteAccess
; CHECK:   [n] -> { Stmt_for_body[i0] -> MemRef_A[o0] };
