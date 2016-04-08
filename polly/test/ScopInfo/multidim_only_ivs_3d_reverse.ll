; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; This test case checks for array access functions where the order in which the
; loop ivs appear in the array subscript does not follow the order of the
; the loops in which they are defined. This (very common) case caused problems
; in the delinearization pass.
;
; void foo(long n, long m, long o, double A[n][m][o]) {
;
;   for (long i = 0; i < n; i++)
;     for (long k = 0; k < o; k++)
;       for (long j = 0; j < m; j++)
;         A[i][j][k] = 1.0;
; }

; CHECK:      Assumed Context:
; CHECK-NEXT: [m, o, n] -> {  :  }
;
; CHECK:      p0: %m
; CHECK-NEXT: p1: %o
; CHECK-NEXT: p2: %n
; CHECK-NOT:  p3
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_j
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [m, o, n] -> { Stmt_for_j[i0, i1, i2] : 0 <= i0 < n and 0 <= i1 < o and 0 <= i2 < m };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [m, o, n] -> { Stmt_for_j[i0, i1, i2] -> [i0, i1, i2] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [m, o, n] -> { Stmt_for_j[i0, i1, i2] -> MemRef_A[i0, i2, i1] };
; CHECK-NEXT: }

define void @foo(i64 %n, i64 %m, i64 %o, double* %A) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.i ], [ %k.inc, %for.k.inc ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.k ], [ %j.inc, %for.j.inc ]
  %subscript0 = mul i64 %i, %m
  %subscript1 = add i64 %j, %subscript0
  %subscript2 = mul i64 %subscript1, %o
  %subscript = add i64 %subscript2, %k
  %idx = getelementptr inbounds double, double* %A, i64 %subscript
  store double 1.0, double* %idx
  br label %for.j.inc

for.j.inc:
  %j.inc = add nsw i64 %j, 1
  %j.exitcond = icmp eq i64 %j.inc, %m
  br i1 %j.exitcond, label %for.k.inc, label %for.j

for.k.inc:
  %k.inc = add nsw i64 %k, 1
  %k.exitcond = icmp eq i64 %k.inc, %o
  br i1 %k.exitcond, label %for.i.inc, label %for.k

for.i.inc:
  %i.inc = add nsw i64 %i, 1
  %i.exitcond = icmp eq i64 %i.inc, %n
  br i1 %i.exitcond, label %end, label %for.i

end:
  ret void
}
