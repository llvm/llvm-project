; RUN: opt %loadPolly -polly-ast -analyze -polly-precise-inbounds < %s \
; RUN:                -polly-precise-fold-accesses \
; RUN:   | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; void foo(long n, long m, int o, double A[n][m], long p, long q) {
;
; if (o >= 0)
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;         A[i+p][j+q] = 1.0;
; else
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;         A[i+p][j+q-100] = 1.0;
;

; This test case is meant to verify that the run-time condition generated
; for the delinearization is simplified such that conditions that would not
; cause any code to be executed are not generated.

; CHECK: if (((o >= 1 && q <= 0 && m + q >= 0) || (o <= 0 && m + q >= 100 && q <= 100)) && 0 == ((m >= 1 && n + p >= 9223372036854775809) || (o <= 0 && n >= 1 && m + q >= 9223372036854775909) || (o <= 0 && m >= 1 && n >= 1 && q <= -9223372036854775709)))

; CHECK:     if (o <= 0) {
; CHECK:       for (int c0 = 0; c0 < n; c0 += 1)
; CHECK:         for (int c1 = 0; c1 < m; c1 += 1)
; CHECK:           Stmt_for_j_1(c0, c1);
; CHECK:     } else
; CHECK:       for (int c0 = 0; c0 < n; c0 += 1)
; CHECK:         for (int c1 = 0; c1 < m; c1 += 1)
; CHECK:           Stmt_for_j(c0, c1);

; CHECK: else
; CHECK:     {  /* original code */ }

define void @foo(i64 %n, i64 %m, i64 %o, double* %A, i64 %p, i64 %q) {
entry:
  br label %cond

cond:
  %cmp = icmp sgt i64 %o, 0
  br i1 %cmp, label %for.i, label %for.i.1

for.i:
  %i = phi i64 [ 0, %cond ], [ %i.inc, %for.i.inc ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %for.j.inc ]
  %offset0 = add nsw i64 %i, %p
  %subscript0 = mul i64 %offset0, %m
  %offset1 = add nsw i64 %j, %q
  %subscript1 = add i64 %offset1, %subscript0
  %idx = getelementptr inbounds double, double* %A, i64 %subscript1
  store double 1.0, double* %idx
  br label %for.j.inc

for.j.inc:
  %j.inc = add nsw i64 %j, 1
  %j.exitcond = icmp eq i64 %j.inc, %m
  br i1 %j.exitcond, label %for.i.inc, label %for.j

for.i.inc:
  %i.inc = add nsw i64 %i, 1
  %i.exitcond = icmp eq i64 %i.inc, %n
  br i1 %i.exitcond, label %end, label %for.i

for.i.1:
  %i.1 = phi i64 [ 0, %cond ], [ %i.inc.1, %for.i.inc.1 ]
  br label %for.j.1

for.j.1:
  %j.1 = phi i64 [ 0, %for.i.1 ], [ %j.inc.1, %for.j.inc.1 ]
  %offset0.1 = add nsw i64 %i.1, %p
  %subscript0.1 = mul i64 %offset0.1, %m
  %offset1.1 = add nsw i64 %j.1, %q
  %subscript1.1 = add i64 %offset1.1, %subscript0.1
  %subscript1.2 = sub i64 %subscript1.1, 100
  %idx.1 = getelementptr inbounds double, double* %A, i64 %subscript1.2
  store double 1.0, double* %idx.1
  br label %for.j.inc.1

for.j.inc.1:
  %j.inc.1 = add nsw i64 %j.1, 1
  %j.exitcond.1 = icmp eq i64 %j.inc.1, %m
  br i1 %j.exitcond.1, label %for.i.inc.1, label %for.j.1

for.i.inc.1:
  %i.inc.1 = add nsw i64 %i.1, 1
  %i.exitcond.1 = icmp eq i64 %i.inc.1, %n
  br i1 %i.exitcond.1, label %end, label %for.i.1

end:
  ret void
}
