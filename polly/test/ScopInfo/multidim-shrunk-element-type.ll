; RUN: opt %loadNPMPolly -polly-only-func=shrink_after_sizes \
; RUN:     '-passes=polly-custom<scops>' -polly-print-scops -disable-output \
; RUN:     < %s 2>&1 | FileCheck %s --check-prefix=AFTER
; RUN: opt %loadNPMPolly -polly-only-func=sizes_after_shrink \
; RUN:     '-passes=polly-custom<scops>' -polly-print-scops -disable-output \
; RUN:     < %s 2>&1 | FileCheck %s --check-prefix=BEFORE
; RUN: opt %loadNPMPolly -polly-only-func=three_dimensions \
; RUN:     '-passes=polly-custom<scops>' -polly-print-scops -disable-output \
; RUN:     < %s 2>&1 | FileCheck %s --check-prefix=THREE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; A memset shrinks the canonical element type of A from double to i8. Dimension
; sizes and subscripts both count canonical elements, so the innermost of each
; has to grow by the same factor. Left in double units, a row would hold m
; bytes rather than 8m, the stores would be modelled as running past its end,
; and the resulting inbounds assumption would be infeasible and drop the SCoP.

; The memset follows the loop nest, so the sizes are recorded while double is
; still the canonical element type and have to be restated when it shrinks.
;
; void shrink_after_sizes(long n, long m, double A[n][m]) {
;   for (long i = 0; i < 100; i++)
;     for (long j = 0; j < 150; j++)
;       A[i][j] = 1.0;
;   memset(A, 0, m * sizeof(double));
; }

; AFTER:      Assumed Context:
; AFTER-NEXT: [m] -> {  : m >= 150 }

; AFTER:      Arrays {
; AFTER-NEXT:     i8 MemRef_A[*][(8 * %m)]; // Element size 1
; AFTER-NEXT: }

; AFTER:      Statements {
; AFTER-NEXT:     Stmt_for_j
; AFTER:              MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; AFTER-NEXT:             [m] -> { Stmt_for_j[i0, i1] -> MemRef_A[i0, o1] : 8i1 <= o1 <= 7 + 8i1 };
; AFTER-NEXT:     Stmt_memset_bb
; AFTER:              MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; AFTER-NEXT:             [m] -> { Stmt_memset_bb[] -> MemRef_A[0, o1] : 0 <= o1 < 8m };
; AFTER-NEXT: }

define void @shrink_after_sizes(i64 %n, i64 %m, ptr %A) {
entry:
  %len = shl nsw i64 %m, 3
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  %tmp = mul nsw i64 %i, %m
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %for.j ]
  %vlaarrayidx.sum = add i64 %j, %tmp
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %vlaarrayidx.sum
  store double 1.0, ptr %arrayidx
  %j.inc = add nsw i64 %j, 1
  %j.exitcond = icmp eq i64 %j.inc, 150
  br i1 %j.exitcond, label %for.i.inc, label %for.j

for.i.inc:
  %i.inc = add nsw i64 %i, 1
  %i.exitcond = icmp eq i64 %i.inc, 100
  br i1 %i.exitcond, label %memset.bb, label %for.i

memset.bb:
  call void @llvm.memset.p0.i64(ptr %A, i8 0, i64 %len, i1 false)
  br label %end

end:
  ret void
}

; The memset comes first, so the array is created with i8 as its element type
; and the sizes of the loop nest arrive afterwards, stated in doubles. They
; have to be restated before they are compared against the ones on record.
;
; void sizes_after_shrink(long n, long m, double A[n][m]) {
;   for (long i = 0; i < 100; i++) {
;     memset(A, 0, m * sizeof(double));
;     for (long j = 0; j < 150; j++)
;       A[i][j] = 1.0;
;   }
; }

; BEFORE:      Assumed Context:
; BEFORE-NEXT: [m] -> {  : m >= 150 }

; BEFORE:      Arrays {
; BEFORE-NEXT:     i8 MemRef_A[*][(8 * %m)]; // Element size 1
; BEFORE-NEXT: }

; BEFORE:      Statements {
; BEFORE-NEXT:     Stmt_memset_bb
; BEFORE:              MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; BEFORE-NEXT:             [m] -> { Stmt_memset_bb[i0] -> MemRef_A[0, o1] : 0 <= o1 < 8m };
; BEFORE-NEXT:     Stmt_for_j
; BEFORE:              MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; BEFORE-NEXT:             [m] -> { Stmt_for_j[i0, i1] -> MemRef_A[i0, o1] : 8i1 <= o1 <= 7 + 8i1 };
; BEFORE-NEXT: }

define void @sizes_after_shrink(i64 %n, i64 %m, ptr %A) {
entry:
  %len = shl nsw i64 %m, 3
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  %tmp = mul nsw i64 %i, %m
  br label %memset.bb

memset.bb:
  call void @llvm.memset.p0.i64(ptr %A, i8 0, i64 %len, i1 false)
  br label %for.j

for.j:
  %j = phi i64 [ 0, %memset.bb ], [ %j.inc, %for.j ]
  %vlaarrayidx.sum = add i64 %j, %tmp
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %vlaarrayidx.sum
  store double 1.0, ptr %arrayidx
  %j.inc = add nsw i64 %j, 1
  %j.exitcond = icmp eq i64 %j.inc, 150
  br i1 %j.exitcond, label %for.i.inc, label %for.j

for.i.inc:
  %i.inc = add nsw i64 %i, 1
  %i.exitcond = icmp eq i64 %i.inc, 100
  br i1 %i.exitcond, label %end, label %for.i

end:
  ret void
}

; Only the innermost dimension is stretched. The outer ones count rows, and a
; row grows together with the innermost dimension, so q stays as it is while r
; becomes 8r.
;
; void three_dimensions(long q, long r, double A[100][q][r]) {
;   for (long i = 0; i < 100; i++)
;     for (long j = 0; j < 150; j++)
;       for (long k = 0; k < 200; k++)
;         A[i][j][k] = 1.0;
;   memset(A, 0, r * sizeof(double));
; }

; THREE:      Assumed Context:
; THREE-NEXT: [q, r] -> {  : q >= 150 and r >= 200 }

; THREE:      Arrays {
; THREE-NEXT:     i8 MemRef_A[*][%q][(8 * %r)]; // Element size 1
; THREE-NEXT: }

; THREE:      Statements {
; THREE-NEXT:     Stmt_for_k
; THREE:              MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; THREE-NEXT:             [q, r] -> { Stmt_for_k[i0, i1, i2] -> MemRef_A[i0, i1, o2] : 8i2 <= o2 <= 7 + 8i2 };

define void @three_dimensions(i64 %q, i64 %r, ptr %A) {
entry:
  %len = shl nsw i64 %r, 3
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  %t1 = mul nsw i64 %i, %q
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %for.j.inc ]
  %t2 = add nsw i64 %t1, %j
  %t3 = mul nsw i64 %t2, %r
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.j ], [ %k.inc, %for.k ]
  %idx = add nsw i64 %t3, %k
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idx
  store double 1.0, ptr %arrayidx
  %k.inc = add nsw i64 %k, 1
  %k.exitcond = icmp eq i64 %k.inc, 200
  br i1 %k.exitcond, label %for.j.inc, label %for.k

for.j.inc:
  %j.inc = add nsw i64 %j, 1
  %j.exitcond = icmp eq i64 %j.inc, 150
  br i1 %j.exitcond, label %for.i.inc, label %for.j

for.i.inc:
  %i.inc = add nsw i64 %i, 1
  %i.exitcond = icmp eq i64 %i.inc, 100
  br i1 %i.exitcond, label %memset.bb, label %for.i

memset.bb:
  call void @llvm.memset.p0.i64(ptr %A, i8 0, i64 %len, i1 false)
  br label %end

end:
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1)
