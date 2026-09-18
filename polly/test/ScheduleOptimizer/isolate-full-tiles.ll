; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -polly-print-ast \
; RUN:     -disable-output < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: opt %loadNPMPolly -polly-isolate-full-tiles \
; RUN:     '-passes=polly-custom<opt-isl;ast>' -polly-print-ast \
; RUN:     -disable-output < %s | FileCheck %s --check-prefix=ALL
; RUN: opt %loadNPMPolly -polly-isolate-full-tiles \
; RUN:     -polly-isolate-complete-tile-dims=1 \
; RUN:     '-passes=polly-custom<opt-isl;ast>' -polly-print-ast \
; RUN:     -disable-output < %s | FileCheck %s --check-prefix=INNERMOST
;
;    void foo(float *A, float *B) {
;      for (long i = 0; i < 100; i++)
;        for (long j = 0; j < 100; j++)
;          A[100 * i + j] = B[100 * i + j] + 1;
;    }
;
; 100 iterations tiled by 32 leave a partial tile of four in each dimension.
;
; Without isolation there is a single copy of the loop nest and both point
; loops carry a min() upper bound.
;
; DEFAULT:      // 1st level tiling - Tiles
; DEFAULT-NEXT: for (int c0 = 0; c0 <= 3; c0 += 1)
; DEFAULT-NEXT:   for (int c1 = 0; c1 <= 3; c1 += 1) {
; DEFAULT-NEXT:     // 1st level tiling - Points
; DEFAULT-NEXT:     for (int c2 = 0; c2 <= min(31, -32 * c0 + 99); c2 += 1)
; DEFAULT-NEXT:       for (int c3 = 0; c3 <= min(31, -32 * c1 + 99); c3 += 1)
; DEFAULT-NEXT:         Stmt_for_j(32 * c0 + c2, 32 * c1 + c3);
; DEFAULT-NOT:  // 1st level tiling - Points
;
; With isolation the complete tiles are separated from the two tails. The nine
; complete tiles have constant bounds in both point loops.
;
; ALL:      // 1st level tiling - Tiles
; ALL:      for (int c0 = 0; c0 <= 2; c0 += 1) {
; ALL-NEXT:   for (int c1 = 0; c1 <= 2; c1 += 1) {
; ALL-NEXT:     // 1st level tiling - Points
; ALL-NEXT:     for (int c2 = 0; c2 <= 31; c2 += 1)
; ALL-NEXT:       for (int c3 = 0; c3 <= 31; c3 += 1)
; ALL-NEXT:         Stmt_for_j(32 * c0 + c2, 32 * c1 + c3);
;
; The tail of the second dimension, columns 96 to 99, runs once per complete
; tile row and keeps its constant bound in the first dimension.
;
; ALL:      // 1st level tiling - Points
; ALL-NEXT: for (int c2 = 0; c2 <= 31; c2 += 1)
; ALL-NEXT:   for (int c3 = 0; c3 <= 3; c3 += 1)
; ALL-NEXT:     Stmt_for_j(32 * c0 + c2, c3 + 96);
;
; The tail of the first dimension, rows 96 to 99, spans the whole width and is
; the only part left with a min() bound.
;
; ALL:      for (int c1 = 0; c1 <= 3; c1 += 1) {
; ALL-NEXT:   // 1st level tiling - Points
; ALL-NEXT:   for (int c2 = 0; c2 <= 3; c2 += 1)
; ALL-NEXT:     for (int c3 = 0; c3 <= min(31, -32 * c1 + 99); c3 += 1)
; ALL-NEXT:       Stmt_for_j(c2 + 96, 32 * c1 + c3);
; ALL-NOT:  // 1st level tiling - Points
;
; Requiring only the innermost dimension to be complete separates along that
; dimension alone. The outer tile loop is not split, so the nest is copied
; twice instead of three times and only the innermost point loop gets a
; constant bound.
;
; INNERMOST:      // 1st level tiling - Tiles
; INNERMOST-NEXT: for (int c0 = 0; c0 <= 3; c0 += 1) {
; INNERMOST-NEXT:   for (int c1 = 0; c1 <= 2; c1 += 1) {
; INNERMOST-NEXT:     // 1st level tiling - Points
; INNERMOST-NEXT:     for (int c2 = 0; c2 <= min(31, -32 * c0 + 99); c2 += 1)
; INNERMOST-NEXT:       for (int c3 = 0; c3 <= 31; c3 += 1)
; INNERMOST-NEXT:         Stmt_for_j(32 * c0 + c2, 32 * c1 + c3);
;
; Its tail keeps the min() bound of the dimension that was left alone.
;
; INNERMOST:      // 1st level tiling - Points
; INNERMOST-NEXT: for (int c2 = 0; c2 <= min(31, -32 * c0 + 99); c2 += 1)
; INNERMOST-NEXT:   for (int c3 = 0; c3 <= 3; c3 += 1)
; INNERMOST-NEXT:     Stmt_for_j(32 * c0 + c2, c3 + 96);
; INNERMOST-NOT:  // 1st level tiling - Points

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %A, ptr %B) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.i.inc ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.next, %for.j ]
  %mul = mul nuw nsw i64 %i, 100
  %idx = add nuw nsw i64 %mul, %j
  %ptrB = getelementptr inbounds float, ptr %B, i64 %idx
  %valB = load float, ptr %ptrB
  %add = fadd float %valB, 1.000000e+00
  %ptrA = getelementptr inbounds float, ptr %A, i64 %idx
  store float %add, ptr %ptrA
  %j.next = add nuw nsw i64 %j, 1
  %j.cmp = icmp eq i64 %j.next, 100
  br i1 %j.cmp, label %for.i.inc, label %for.j

for.i.inc:
  %i.next = add nuw nsw i64 %i, 1
  %i.cmp = icmp eq i64 %i.next, 100
  br i1 %i.cmp, label %exit, label %for.i

exit:
  ret void
}
