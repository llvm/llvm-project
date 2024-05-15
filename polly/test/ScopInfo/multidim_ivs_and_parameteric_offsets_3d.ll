; RUN: opt %loadPolly -polly-precise-fold-accesses -polly-print-scops -disable-output < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; void foo(long n, long m, long o, double A[n][m][o], long p, long q, long r) {
;
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       for (long k = 0; k < o; k++)
;         A[i+p][j+q][k+r] = 1.0;
; }
;
; Access function:
;    {{{((8 * ((((%m * %p) + %q) * %o) + %r)) + %A),+,(8 * %m * %o)}<%for.i>,+,
;        (8 * %o)}<%for.j>,+,8}<%for.k>

; CHECK:      Assumed Context:
; CHECK-NEXT: [o, m, n, p, q, r] -> {  : -m <= q <= 1 and ((-m < q <= 0 and -o < r < 0) or (r = 0 and q <= 0) or (r = -o and q > -m)) }
;
; CHECK:      p0: %o
; CHECK-NEXT: p1: %m
; CHECK-NEXT: p2: %n
; CHECK-NEXT: p3: %p
; CHECK-NEXT: p4: %q
; CHECK-NEXT: p5: %r
; CHECK-NOT:  p6
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_k
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [o, m, n, p, q, r] -> { Stmt_for_k[i0, i1, i2] : 0 <= i0 < n and 0 <= i1 < m and 0 <= i2 < o };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [o, m, n, p, q, r] -> { Stmt_for_k[i0, i1, i2] -> [i0, i1, i2] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [o, m, n, p, q, r] -> { Stmt_for_k[i0, i1, i2] -> MemRef_A[-1 + p + i0, -1 + m + q + i1, o + r + i2] : i1 <= -q and i2 < -r; Stmt_for_k[i0, i1, i2] -> MemRef_A[p + i0, -1 + q + i1, o + r + i2] : i1 > -q and i2 < -r; Stmt_for_k[i0, i1, i2] -> MemRef_A[-1 + p + i0, m + q + i1, r + i2] : i1 < -q and i2 >= -r; Stmt_for_k[i0, i1, i2] -> MemRef_A[p + i0, q + i1, r + i2] : i1 >= -q and i2 >= -r };
; CHECK-NEXT: }

define void @foo(i64 %n, i64 %m, i64 %o, ptr %A, i64 %p, i64 %q, i64 %r) {
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
  %offset0 = add nsw i64 %i, %p
  %subscript0 = mul i64 %offset0, %m
  %offset1 = add nsw i64 %j, %q
  %subscript1 = add i64 %offset1, %subscript0
  %subscript2 = mul i64 %subscript1, %o
  %offset2 = add nsw i64 %k, %r
  %subscript = add i64 %subscript2, %offset2
  %idx = getelementptr inbounds double, ptr %A, i64 %subscript
  store double 1.0, ptr %idx
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
