; RUN: opt %loadPolly -polly-scops -analyze -polly-allow-nonaffine < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void foo(long n, long m, long o, double A[n][m][o]) {
;
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       for (long k = 0; k < o; k++)
;         A[i][j][k] = 1.0;
; }
;
; Access function: {{{0,+,(%m * %o)}<%for.i>,+,%o}<%for.j>,+,1}<nw><%for.k>
;
; TODO: Recover the multi-dimensional array information to avoid the
;       conservative approximation we are using today.

define void @foo(i64 %n, i64 %m, i64 %o, double* %A) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %for.j.inc ]
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.j ], [ %k.inc, %for.k.inc ]
  %subscript0 = mul i64 %i, %m
  %subscript1 = add i64 %j, %subscript0
  %subscript2 = mul i64 %subscript1, %o
  %subscript = add i64 %subscript2, %k
  %idx = getelementptr inbounds double* %A, i64 %subscript
  store double 1.0, double* %idx
  br label %for.k.inc

for.k.inc:
  %k.inc = add nsw i64 %k, 1
  %k.exitcond = icmp eq i64 %k.inc, %o
  br i1 %k.exitcond, label %for.j.inc, label %for.k

for.j.inc:
  %j.inc = add nsw i64 %j, 1
  %j.exitcond = icmp eq i64 %j.inc, %m
  br i1 %j.exitcond, label %for.i.inc, label %for.j

for.i.inc:
  %i.inc = add nsw i64 %i, 1
  %i.exitcond = icmp eq i64 %i.inc, %n
  br i1 %i.exitcond, label %end, label %for.i

end:
  ret void
}

; CHECK: p0: %n
; CHECK: p1: %m
; CHECK: p2: %o
; CHECK-NOT: p3

; CHECK: Domain
; CHECK:   [n, m, o] -> { Stmt_for_k[i0, i1, i2] : i0 >= 0 and i0 <= -1 + n and i1 >= 0 and i1 <= -1 + m and i2 >= 0 and i2 <= -1 + o };
; CHECK: Scattering
; CHECK:   [n, m, o] -> { Stmt_for_k[i0, i1, i2] -> scattering[0, i0, 0, i1, 0, i2, 0] };
; CHECK: WriteAccess
; CHECK:   [n, m, o] -> { Stmt_for_k[i0, i1, i2] -> MemRef_A[o0] };

