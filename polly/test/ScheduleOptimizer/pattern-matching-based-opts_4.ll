; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -debug < %s 2>&1 | FileCheck %s
; RUN: opt %loadPolly -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 \
; RUN: -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=PATTERN-MATCHING-OPTS
; REQUIRES: asserts
;
;    C := A * B + C
;    Check that the pattern matching optimizations can detect different
;    permutations of GEMM loop and produce the correct ISL AST. In this case,
;    dimensions of band nodes can be implicitly permuted by the algorithm
;    applied during the schedule generation. It should be taken into the
;    account during the pattern matching optimizations.
;    for (i = 0; i < _PB_NI; i++)
;      for (k = 0; k < _PB_NK; ++k)
;        for (j = 0; j < _PB_NJ; j++)
;	   C[i][j] += A[i][k] * B[k][j];
;
; CHECK: The matrix multiplication pattern was detected
;
; PATTERN-MATCHING-OPTS:    // 1st level tiling - Tiles
; PATTERN-MATCHING-OPTS-NEXT:    for (int c1 = 0; c1 <= 3; c1 += 1) {
; PATTERN-MATCHING-OPTS-NEXT:      for (int c3 = 256 * c1; c3 <= 256 * c1 + 255; c3 += 1)
; PATTERN-MATCHING-OPTS-NEXT:        for (int c4 = 0; c4 <= 1023; c4 += 1)
; PATTERN-MATCHING-OPTS-NEXT:          CopyStmt_0(0, c3, c4);
; PATTERN-MATCHING-OPTS-NEXT:      for (int c2 = 0; c2 <= 10; c2 += 1) {
; PATTERN-MATCHING-OPTS-NEXT:        for (int c6 = 96 * c2; c6 <= min(1023, 96 * c2 + 95); c6 += 1)
; PATTERN-MATCHING-OPTS-NEXT:          for (int c7 = 256 * c1; c7 <= 256 * c1 + 255; c7 += 1)
; PATTERN-MATCHING-OPTS-NEXT:            CopyStmt_1(0, c1, c2, c6, c7);
; PATTERN-MATCHING-OPTS-NEXT:        // 1st level tiling - Points
; PATTERN-MATCHING-OPTS-NEXT:        // Register tiling - Tiles
; PATTERN-MATCHING-OPTS-NEXT:        for (int c3 = 0; c3 <= 127; c3 += 1)
; PATTERN-MATCHING-OPTS-NEXT:          for (int c4 = 0; c4 <= min(23, -24 * c2 + 255); c4 += 1)
; PATTERN-MATCHING-OPTS-NEXT:            for (int c5 = 0; c5 <= 255; c5 += 1) {
; PATTERN-MATCHING-OPTS-NEXT:              // Loop Vectorizer Disabled
; PATTERN-MATCHING-OPTS-NEXT:              // Register tiling - Points
; PATTERN-MATCHING-OPTS-NEXT:              {
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 256 * c1 + c5, 8 * c3);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 256 * c1 + c5, 8 * c3 + 1);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 256 * c1 + c5, 8 * c3 + 2);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 256 * c1 + c5, 8 * c3 + 3);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 256 * c1 + c5, 8 * c3 + 4);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 256 * c1 + c5, 8 * c3 + 5);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 256 * c1 + c5, 8 * c3 + 6);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 256 * c1 + c5, 8 * c3 + 7);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 256 * c1 + c5, 8 * c3);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 256 * c1 + c5, 8 * c3 + 1);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 256 * c1 + c5, 8 * c3 + 2);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 256 * c1 + c5, 8 * c3 + 3);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 256 * c1 + c5, 8 * c3 + 4);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 256 * c1 + c5, 8 * c3 + 5);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 256 * c1 + c5, 8 * c3 + 6);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 256 * c1 + c5, 8 * c3 + 7);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 256 * c1 + c5, 8 * c3);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 256 * c1 + c5, 8 * c3 + 1);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 256 * c1 + c5, 8 * c3 + 2);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 256 * c1 + c5, 8 * c3 + 3);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 256 * c1 + c5, 8 * c3 + 4);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 256 * c1 + c5, 8 * c3 + 5);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 256 * c1 + c5, 8 * c3 + 6);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 256 * c1 + c5, 8 * c3 + 7);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 256 * c1 + c5, 8 * c3);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 256 * c1 + c5, 8 * c3 + 1);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 256 * c1 + c5, 8 * c3 + 2);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 256 * c1 + c5, 8 * c3 + 3);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 256 * c1 + c5, 8 * c3 + 4);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 256 * c1 + c5, 8 * c3 + 5);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 256 * c1 + c5, 8 * c3 + 6);
; PATTERN-MATCHING-OPTS-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 256 * c1 + c5, 8 * c3 + 7);
; PATTERN-MATCHING-OPTS-NEXT:              }
; PATTERN-MATCHING-OPTS-NEXT:            }
; PATTERN-MATCHING-OPTS-NEXT:      }
; PATTERN-MATCHING-OPTS-NEXT:    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, double %alpha, double %beta, [1024 x double]* %C, [1024 x double]* %A, [1024 x double]* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc20, %entry.split
  %indvars.iv41 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next42, %for.inc20 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc17, %for.cond1.preheader
  %indvars.iv38 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next39, %for.inc17 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %indvars.iv = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv41, i64 %indvars.iv38
  %tmp = load double, double* %arrayidx8, align 8
  %arrayidx12 = getelementptr inbounds [1024 x double], [1024 x double]* %B, i64 %indvars.iv38, i64 %indvars.iv
  %tmp1 = load double, double* %arrayidx12, align 8
  %mul = fmul double %tmp, %tmp1
  %arrayidx16 = getelementptr inbounds [1024 x double], [1024 x double]* %C, i64 %indvars.iv41, i64 %indvars.iv
  %tmp2 = load double, double* %arrayidx16, align 8
  %add = fadd double %tmp2, %mul
  store double %add, double* %arrayidx16, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body6, label %for.inc17

for.inc17:                                        ; preds = %for.body6
  %indvars.iv.next39 = add nuw nsw i64 %indvars.iv38, 1
  %exitcond40 = icmp ne i64 %indvars.iv.next39, 1024
  br i1 %exitcond40, label %for.cond4.preheader, label %for.inc20

for.inc20:                                        ; preds = %for.inc17
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %exitcond43 = icmp ne i64 %indvars.iv.next42, 1024
  br i1 %exitcond43, label %for.cond1.preheader, label %for.end22

for.end22:                                        ; preds = %for.inc20
  ret void
}
