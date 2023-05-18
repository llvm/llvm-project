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
;  opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
;  -polly-target-throughput-vector-fma=1 \
;  -polly-target-latency-vector-fma=8 \
;  -polly-codegen -polly-target-1st-cache-level-associativity=8 \
;  -polly-target-2nd-cache-level-associativity=8 \
;  -polly-target-1st-cache-level-size=32768 \
;  -polly-target-vector-register-bitwidth=256 \
;  -polly-target-2nd-cache-level-size=262144 -gvn -licm -slp-vectorizer \
;  -mcpu=corei7 -stats -S < %s 2>&1 | FileCheck %s \
; --check-prefix=AUTO-VECTORIZATION
;
;
;    /* We isolate a set of partial tile prefixes, which contains only partial
;       tile prefixes that have exactly Mr x Nr iterations of the two innermost
;       loops produced by the optimization of the matrix multiplication. Mr and
;       Nr are parameters of the micro-kernel (see getMicroKernelParams and
;       getMacroKernelParams from lib/Transform/ScheduleOptimizer.cpp for
;       details). This test check that in case it cannot be proved that
;       the number of loop iterations can be evenly divided by tile sizes
;       and we tile and unroll the point loops, it helps to get rid of
;       the conditional expressions of the unrolled innermost loops, which
;       prevents stores and loads of the unrolled loops from being sunk
;       and hoisted. Otherwise, it causes a run-time regression in comparison
;       to the vectorized code with sunk and hoisted memory accesses. */
;    /* C := A * B + C */
;    for (i = 0; i < 1020; i++)
;      for (j = 0; j < 1020; j++)
;	 for (k = 0; k < 1020; ++k)
;	   C[i][j] += A[i][k] * B[k][j];
;
; CHECK:    // 1st level tiling - Tiles
; CHECK-NEXT:    for (int c1 = 0; c1 <= 3; c1 += 1) {
; CHECK-NEXT:      for (int c3 = 0; c3 <= 1019; c3 += 1)
; CHECK-NEXT:        for (int c4 = 256 * c1; c4 <= min(1019, 256 * c1 + 255); c4 += 1)
; CHECK-NEXT:          CopyStmt_0(0, c3, c4);
; CHECK-NEXT:      for (int c2 = 0; c2 <= 10; c2 += 1) {
; CHECK-NEXT:        for (int c6 = 96 * c2; c6 <= min(1019, 96 * c2 + 95); c6 += 1)
; CHECK-NEXT:          for (int c7 = 256 * c1; c7 <= min(1019, 256 * c1 + 255); c7 += 1)
; CHECK-NEXT:            CopyStmt_1(0, c1, c2, c6, c7);
; CHECK-NEXT:        // 1st level tiling - Points
; CHECK-NEXT:        // Register tiling - Tiles
; CHECK-NEXT:        {
; CHECK-NEXT:          for (int c3 = 0; c3 <= 126; c3 += 1)
; CHECK-NEXT:            for (int c4 = 0; c4 <= min(23, -24 * c2 + 254); c4 += 1)
; CHECK-NEXT:              for (int c5 = 0; c5 <= min(255, -256 * c1 + 1019); c5 += 1) {
; CHECK-NEXT:                // Loop Vectorizer Disabled
; CHECK-NEXT:                // Register tiling - Points
; CHECK-NEXT:                {
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(96 * c2 + 4 * c4 + 3, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                }
; CHECK-NEXT:              }
; CHECK-NEXT:              for (int c4 = 0; c4 <= min(23, -24 * c2 + 254); c4 += 1)
; CHECK-NEXT:                for (int c5 = 0; c5 <= min(255, -256 * c1 + 1019); c5 += 1) {
; CHECK-NEXT:                  // Loop Vectorizer Disabled
; CHECK-NEXT:                  // Register tiling - Points
; CHECK-NEXT:                  {
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4, 1016, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4, 1017, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4, 1018, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4, 1019, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 1, 1016, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 1, 1017, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 1, 1018, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 1, 1019, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 2, 1016, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 2, 1017, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 2, 1018, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 2, 1019, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 3, 1016, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 3, 1017, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 3, 1018, 256 * c1 + c5);
; CHECK-NEXT:                    Stmt_for_body6(96 * c2 + 4 * c4 + 3, 1019, 256 * c1 + c5);
; CHECK-NEXT:                  }
; CHECK-NEXT:                }
; CHECK-NEXT:            }
; CHECK-NEXT:          }
; CHECK-NEXT:        }
;
; AUTO-VECTORIZATION:  fmul <4 x double>
; AUTO-VECTORIZATION:  fadd <4 x double>

; AUTO-VECTORIZATION: 36 SLP              - Number of vector instructions generated
; AUTO-VECTORIZATION: 146 licm             - Number of instructions hoisted out of loop
; AUTO-VECTORIZATION: 1 licm             - Number of load insts hoisted or sunk
; AUTO-VECTORIZATION: 32 licm             - Number of memory locations promoted to registers
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, double %alpha, double %beta, ptr %C, ptr %A, ptr %B) #0 {
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
  %arrayidx8 = getelementptr inbounds [1020 x double], ptr %A, i64 %indvars.iv41, i64 %indvars.iv
  %tmp = load double, ptr %arrayidx8, align 8
  %arrayidx12 = getelementptr inbounds [1020 x double], ptr %B, i64 %indvars.iv, i64 %indvars.iv38
  %tmp1 = load double, ptr %arrayidx12, align 8
  %mul = fmul double %tmp, %tmp1
  %arrayidx16 = getelementptr inbounds [1020 x double], ptr %C, i64 %indvars.iv41, i64 %indvars.iv38
  %tmp2 = load double, ptr %arrayidx16, align 8
  %add = fadd double %tmp2, %mul
  store double %add, ptr %arrayidx16, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1020
  br i1 %exitcond, label %for.body6, label %for.inc17

for.inc17:                                        ; preds = %for.body6
  %indvars.iv.next39 = add nuw nsw i64 %indvars.iv38, 1
  %exitcond40 = icmp ne i64 %indvars.iv.next39, 1020
  br i1 %exitcond40, label %for.cond4.preheader, label %for.inc20

for.inc20:                                        ; preds = %for.inc17
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %exitcond43 = icmp ne i64 %indvars.iv.next42, 1020
  br i1 %exitcond43, label %for.cond1.preheader, label %for.end22

for.end22:                                        ; preds = %for.inc20
  ret void
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+cx16,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt" }
