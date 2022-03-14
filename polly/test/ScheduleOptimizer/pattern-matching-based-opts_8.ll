; RUN: opt %loadPolly -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 \
; RUN: -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s
;
;    /* C := A * B + C */
;    /* Elements of the matrices B, C have the double type. */
;    /* Elements of the matrix A have the float type. */
;    /* The type size of elements of the matrix multiplication operands is used
;       to determine the parameters of the code produced by the optimization
;       of the matrix multiplication (e.g. bounds of the loops of the loop
;       nest, the innermost loop body). This test checks the form of
;       the generated loop nest. See getMicroKernelParams and
;       getMacroKernelParams from lib/Transform/ScheduleOptimizer.cpp
;       for details. */
;    for (i = 0; i < _PB_NI; i++)
;      for (j = 0; j < _PB_NJ; j++)
;	 for (k = 0; k < _PB_NK; ++k)
;	   C[i][j] += A[i][k] * B[k][j];
;
; CHECK:    // 1st level tiling - Tiles
; CHECK-NEXT:    for (int c1 = 0; c1 <= 3; c1 += 1) {
; CHECK-NEXT:      for (int c3 = 0; c3 <= 1023; c3 += 1)
; CHECK-NEXT:        for (int c4 = 256 * c1; c4 <= 256 * c1 + 255; c4 += 1)
; CHECK-NEXT:          CopyStmt_0(0, c3, c4);
; CHECK-NEXT:      for (int c2 = 0; c2 <= 10; c2 += 1) {
; CHECK-NEXT:        for (int c6 = 96 * c2; c6 <= min(1023, 96 * c2 + 95); c6 += 1)
; CHECK-NEXT:          for (int c7 = 256 * c1; c7 <= 256 * c1 + 255; c7 += 1)
; CHECK-NEXT:            CopyStmt_1(0, c1, c2, c6, c7);
; CHECK-NEXT:        // 1st level tiling - Points
; CHECK-NEXT:        // Register tiling - Tiles
; CHECK-NEXT:        for (int c3 = 0; c3 <= 127; c3 += 1)
; CHECK-NEXT:          for (int c4 = 0; c4 <= min(23, -24 * c2 + 255); c4 += 1)
; CHECK-NEXT:            for (int c5 = 0; c5 <= 255; c5 += 1) {
; CHECK-NEXT:              // Loop Vectorizer Disabled
; CHECK-NEXT:              // Register tiling - Points
; CHECK-NEXT:              {
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:              }
; CHECK-NEXT:            }
; CHECK-NEXT:      }
; CHECK-NEXT:    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Function Attrs: noinline nounwind uwtable
define internal void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, double %alpha, double %beta, [1024 x double]* %C, [1024 x float]* %A, [1024 x double]* %B) #0 {
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
  %arrayidx8 = getelementptr inbounds [1024 x float], [1024 x float]* %A, i64 %indvars.iv41, i64 %indvars.iv
  %tmp = load float, float* %arrayidx8, align 4
  %conv = fpext float %tmp to double
  %arrayidx12 = getelementptr inbounds [1024 x double], [1024 x double]* %B, i64 %indvars.iv, i64 %indvars.iv38
  %tmp1 = load double, double* %arrayidx12, align 8
  %mul = fmul double %conv, %tmp1
  %arrayidx16 = getelementptr inbounds [1024 x double], [1024 x double]* %C, i64 %indvars.iv41, i64 %indvars.iv38
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
