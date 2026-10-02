; RUN: opt %loadNPMPolly -polly-isolate-complete-tiles \
; RUN:     -polly-2nd-level-tiling -polly-2nd-level-default-tile-size=12 \
; RUN:     '-passes=polly-custom<opt-isl;ast>' -polly-print-ast \
; RUN:     -disable-output < %s | FileCheck %s --check-prefix=L2
; RUN: opt %loadNPMPolly -polly-isolate-complete-tiles \
; RUN:     -polly-2nd-level-tiling -polly-2nd-level-default-tile-size=12 \
; RUN:     -polly-isolate-complete-tiles-2nd-level \
; RUN:     '-passes=polly-custom<opt-isl;ast>' -polly-print-ast \
; RUN:     -disable-output < %s | FileCheck %s --check-prefix=L2-ISO
; RUN: opt %loadNPMPolly -polly-isolate-complete-tiles \
; RUN:     -polly-register-tiling -polly-register-tiling-default-tile-size=3 \
; RUN:     '-passes=polly-custom<opt-isl;ast>' -polly-print-ast \
; RUN:     -disable-output < %s | FileCheck %s --check-prefix=REG
; RUN: opt %loadNPMPolly -polly-isolate-complete-tiles \
; RUN:     -polly-register-tiling -polly-register-tiling-default-tile-size=3 \
; RUN:     -polly-isolate-complete-register-tiles \
; RUN:     '-passes=polly-custom<opt-isl;ast>' -polly-print-ast \
; RUN:     -disable-output < %s | FileCheck %s --check-prefix=REG-ISO
;
;    void foo(float *A, float *B) {
;      for (long i = 0; i < 100; i++)
;        for (long j = 0; j < 100; j++)
;          A[100 * i + j] = B[100 * i + j] + 1;
;    }
;
; A complete first-level tile is 32 wide, which is not a multiple of the inner
; tile sizes used here, 12 for the second level and 3 for register tiling. The
; inner tiles of a complete first-level tile are therefore not all complete,
; and isolating the first level alone does not give their point loops constant
; bounds.
;
; Without the second-level option, the second-level point loops keep a min()
; bound inside the complete first-level tiles.
;
; L2:      // 1st level tiling - Tiles
; L2:      for (int c0 = 0; c0 <= 2; c0 += 1) {
; L2-NEXT:   for (int c1 = 0; c1 <= 2; c1 += 1) {
; L2-NEXT:     // 1st level tiling - Points
; L2-NEXT:     // 2nd level tiling - Tiles
; L2-NEXT:     for (int c2 = 0; c2 <= 2; c2 += 1)
; L2-NEXT:       for (int c3 = 0; c3 <= 2; c3 += 1) {
; L2-NEXT:         // 2nd level tiling - Points
; L2-NEXT:         for (int c4 = 0; c4 <= min(11, -12 * c2 + 31); c4 += 1)
; L2-NEXT:           for (int c5 = 0; c5 <= min(11, -12 * c3 + 31); c5 += 1)
; L2-NEXT:             Stmt_for_j(32 * c0 + 12 * c2 + c4, 32 * c1 + 12 * c3 + c5);
;
; With it, the two by two complete second-level tiles have constant bounds and
; the remainder of eight is separated from them.
;
; L2-ISO:      // 1st level tiling - Tiles
; L2-ISO:      for (int c0 = 0; c0 <= 2; c0 += 1) {
; L2-ISO-NEXT:   for (int c1 = 0; c1 <= 2; c1 += 1) {
; L2-ISO-NEXT:     // 1st level tiling - Points
; L2-ISO-NEXT:     // 2nd level tiling - Tiles
; L2-ISO-NEXT:     {
; L2-ISO-NEXT:       for (int c2 = 0; c2 <= 1; c2 += 1) {
; L2-ISO-NEXT:         for (int c3 = 0; c3 <= 1; c3 += 1) {
; L2-ISO-NEXT:           // 2nd level tiling - Points
; L2-ISO-NEXT:           for (int c4 = 0; c4 <= 11; c4 += 1)
; L2-ISO-NEXT:             for (int c5 = 0; c5 <= 11; c5 += 1)
; L2-ISO-NEXT:               Stmt_for_j(32 * c0 + 12 * c2 + c4, 32 * c1 + 12 * c3 + c5);
; L2-ISO-NEXT:         }
; L2-ISO-NEXT:         // 2nd level tiling - Points
; L2-ISO-NEXT:         for (int c4 = 0; c4 <= 11; c4 += 1)
; L2-ISO-NEXT:           for (int c5 = 0; c5 <= 7; c5 += 1)
; L2-ISO-NEXT:             Stmt_for_j(32 * c0 + 12 * c2 + c4, 32 * c1 + c5 + 24);
;
; Without the register option, the unrolled body of a register tile guards the
; statements that fall into the remainder of two.
;
; REG:      // 1st level tiling - Tiles
; REG:      for (int c0 = 0; c0 <= 2; c0 += 1) {
; REG-NEXT:   for (int c1 = 0; c1 <= 2; c1 += 1) {
; REG-NEXT:     // 1st level tiling - Points
; REG-NEXT:     // Register tiling - Tiles
; REG-NEXT:     for (int c2 = 0; c2 <= 10; c2 += 1)
; REG-NEXT:       for (int c3 = 0; c3 <= 10; c3 += 1) {
; REG-NEXT:         // Register tiling - Points
; REG-NEXT:         {
; REG-NEXT:           Stmt_for_j(32 * c0 + 3 * c2, 32 * c1 + 3 * c3);
; REG-NEXT:           Stmt_for_j(32 * c0 + 3 * c2, 32 * c1 + 3 * c3 + 1);
; REG-NEXT:           if (c3 <= 9)
; REG-NEXT:             Stmt_for_j(32 * c0 + 3 * c2, 32 * c1 + 3 * c3 + 2);
;
; With it, the complete register tiles are unrolled without any guard.
;
; REG-ISO:      // 1st level tiling - Tiles
; REG-ISO:      for (int c0 = 0; c0 <= 2; c0 += 1) {
; REG-ISO-NEXT:   for (int c1 = 0; c1 <= 2; c1 += 1) {
; REG-ISO-NEXT:     // 1st level tiling - Points
; REG-ISO-NEXT:     // Register tiling - Tiles
; REG-ISO-NEXT:     {
; REG-ISO-NEXT:       for (int c2 = 0; c2 <= 9; c2 += 1) {
; REG-ISO-NEXT:         for (int c3 = 0; c3 <= 9; c3 += 1) {
; REG-ISO-NEXT:           // Register tiling - Points
; REG-ISO-NEXT:           {
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2, 32 * c1 + 3 * c3);
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2, 32 * c1 + 3 * c3 + 1);
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2, 32 * c1 + 3 * c3 + 2);
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2 + 1, 32 * c1 + 3 * c3);
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2 + 1, 32 * c1 + 3 * c3 + 1);
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2 + 1, 32 * c1 + 3 * c3 + 2);
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2 + 2, 32 * c1 + 3 * c3);
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2 + 2, 32 * c1 + 3 * c3 + 1);
; REG-ISO-NEXT:             Stmt_for_j(32 * c0 + 3 * c2 + 2, 32 * c1 + 3 * c3 + 2);
; REG-ISO-NEXT:           }

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
