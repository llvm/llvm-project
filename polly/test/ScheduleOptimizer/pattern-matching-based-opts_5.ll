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
; opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; -polly-target-throughput-vector-fma=1 \
; -polly-target-latency-vector-fma=8 \
; -polly-codegen -polly-target-1st-cache-level-associativity=8 \
; -polly-target-2nd-cache-level-associativity=8 \
; -polly-target-1st-cache-level-size=32768 \
; -polly-target-vector-register-bitwidth=256 \
; -polly-target-2nd-cache-level-size=262144 -gvn -licm -slp-vectorizer \
; -mcpu=corei7 -stats -S < %s 2>&1 | FileCheck %s --check-prefix=AUTO-VECTORIZATION
;
;
;    /* We isolate a set of partial tile prefixes, which contains only partial
;       tile prefixes that have exactly Mr x Nr iterations of the two innermost
;       loops produced by the optimization of the matrix multiplication. Mr and
;       Nr are parameters of the micro-kernel (see getMicroKernelParams and
;       getMacroKernelParams from lib/Transform/ScheduleOptimizer.cpp for
;       details). This test check that in case of parametric bounds it helps to
;       get rid of the conditional expressions of the unrolled innermost loops,
;       which prevents stores and loads of the unrolled loops from being sunk
;       and hoisted. Otherwise, it causes a run-time regression in comparison
;       to the vectorized code with sunk and hoisted memory accesses. */
;
;    /* C := A * B + C */
;    for (i = 0; i < _PB_NI; i++)
;      for (j = 0; j < _PB_NJ; j++)
;	 for (k = 0; k < _PB_NK; ++k)
;	   C[i][j] += A[i][k] * B[k][j];
;
; CHECK:          if (ni >= 1) {
; CHECK-NEXT:            // 1st level tiling - Tiles
; CHECK-NEXT:            for (int c0 = 0; c0 <= floord(nj - 1, 2048); c0 += 1)
; CHECK-NEXT:              for (int c1 = 0; c1 <= floord(nk - 1, 256); c1 += 1) {
; CHECK-NEXT:                for (int c3 = 2048 * c0; c3 <= min(nj - 1, 2048 * c0 + 2047); c3 += 1)
; CHECK-NEXT:                  for (int c4 = 256 * c1; c4 <= min(nk - 1, 256 * c1 + 255); c4 += 1)
; CHECK-NEXT:                    CopyStmt_0(0, c3, c4);
; CHECK-NEXT:                for (int c2 = 0; c2 <= floord(ni - 1, 96); c2 += 1) {
; CHECK-NEXT:                    for (int c6 = 96 * c2; c6 <= min(ni - 1, 96 * c2 + 95); c6 += 1)
; CHECK-NEXT:                      for (int c7 = 256 * c1; c7 <= min(nk - 1, 256 * c1 + 255); c7 += 1)
; CHECK-NEXT:                        CopyStmt_1(c0, c1, c2, c6, c7);
; CHECK-NEXT:                  // 1st level tiling - Points
; CHECK-NEXT:                  // Register tiling - Tiles
; CHECK-NEXT:                  {
; CHECK-NEXT:                    if (ni >= 96 * c2 + 4)
; CHECK-NEXT:                      for (int c3 = 0; c3 <= min(255, -256 * c0 + nj / 8 - 1); c3 += 1) {
; CHECK-NEXT:                        for (int c4 = 0; c4 <= min(23, -24 * c2 + ni / 4 - 1); c4 += 1)
; CHECK-NEXT:                          for (int c5 = 0; c5 <= min(255, nk - 256 * c1 - 1); c5 += 1) {
; CHECK-NEXT:                            // Loop Vectorizer Disabled
; CHECK-NEXT:                            // Register tiling - Points
; CHECK-NEXT:                            {
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                            }
; CHECK-NEXT:                          }
; CHECK-NEXT:                        if ((ni >= 96 * c2 + 5 && 96 * c2 + 7 >= ni && c3 >= 0) || (ni >= 96 * c2 + 8 && 96 * c2 + 95 >= ni && ni % 4 >= 1)) {
; CHECK-NEXT:                          if (96 * c2 + 7 >= ni) {
; CHECK-NEXT:                            for (int c5 = 0; c5 <= min(255, nk - 256 * c1 - 1); c5 += 1) {
; CHECK-NEXT:                              // Loop Vectorizer Disabled
; CHECK-NEXT:                              // Register tiling - Points
; CHECK-NEXT:                              {
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                if (ni >= 96 * c2 + 6) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 5, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 5, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 5, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 5, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 5, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 5, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 5, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 5, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                  if (96 * c2 + 7 == ni) {
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                              }
; CHECK-NEXT:                            }
; CHECK-NEXT:                          } else {
; CHECK-NEXT:                            for (int c5 = 0; c5 <= min(255, nk - 256 * c1 - 1); c5 += 1) {
; CHECK-NEXT:                              // Loop Vectorizer Disabled
; CHECK-NEXT:                              // Register tiling - Points
; CHECK-NEXT:                              {
; CHECK-NEXT:                                Stmt_for_body6(-((ni - 1) % 4) + ni - 1, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(-((ni - 1) % 4) + ni - 1, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(-((ni - 1) % 4) + ni - 1, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(-((ni - 1) % 4) + ni - 1, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(-((ni - 1) % 4) + ni - 1, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(-((ni - 1) % 4) + ni - 1, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(-((ni - 1) % 4) + ni - 1, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                Stmt_for_body6(-((ni - 1) % 4) + ni - 1, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                if (ni % 4 >= 2) {
; CHECK-NEXT:                                  Stmt_for_body6(-((ni - 1) % 4) + ni, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(-((ni - 1) % 4) + ni, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(-((ni - 1) % 4) + ni, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(-((ni - 1) % 4) + ni, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(-((ni - 1) % 4) + ni, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(-((ni - 1) % 4) + ni, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(-((ni - 1) % 4) + ni, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                  Stmt_for_body6(-((ni - 1) % 4) + ni, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                  if ((ni + 1) % 4 == 0) {
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                              }
; CHECK-NEXT:                            }
; CHECK-NEXT:                          }
; CHECK-NEXT:                        }
; CHECK-NEXT:                      }
; CHECK-NEXT:                    if (96 * c2 + 3 >= ni || 2048 * c0 + 7 >= nj || (2048 * c0 + 2047 >= nj && nj % 8 >= 1)) {
; CHECK-NEXT:                      if (2048 * c0 + 7 >= nj) {
; CHECK-NEXT:                        for (int c4 = 0; c4 <= min(23, -24 * c2 + (ni - 1) / 4); c4 += 1)
; CHECK-NEXT:                          for (int c5 = 0; c5 <= min(255, nk - 256 * c1 - 1); c5 += 1) {
; CHECK-NEXT:                            // Loop Vectorizer Disabled
; CHECK-NEXT:                            // Register tiling - Points
; CHECK-NEXT:                            {
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0, 256 * c1 + c5);
; CHECK-NEXT:                              if (nj >= 2048 * c0 + 2) {
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                if (nj >= 2048 * c0 + 3) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj >= 2048 * c0 + 4) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj >= 2048 * c0 + 5) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj >= 2048 * c0 + 6) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 4 * c4, 2048 * c0 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                        if (2048 * c0 + 7 == nj)
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 4 * c4, nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                              }
; CHECK-NEXT:                              if (ni >= 96 * c2 + 4 * c4 + 2) {
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0, 256 * c1 + c5);
; CHECK-NEXT:                                if (nj >= 2048 * c0 + 2) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj >= 2048 * c0 + 3) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj >= 2048 * c0 + 4) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj >= 2048 * c0 + 5) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj >= 2048 * c0 + 6) {
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 4 * c4 + 1, 2048 * c0 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                          if (2048 * c0 + 7 == nj)
; CHECK-NEXT:                                            Stmt_for_body6(96 * c2 + 4 * c4 + 1, nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                                if (ni >= 96 * c2 + 4 * c4 + 3) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj >= 2048 * c0 + 2) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj >= 2048 * c0 + 3) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj >= 2048 * c0 + 4) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj >= 2048 * c0 + 5) {
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                          if (nj >= 2048 * c0 + 6) {
; CHECK-NEXT:                                            Stmt_for_body6(96 * c2 + 4 * c4 + 2, 2048 * c0 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                            if (2048 * c0 + 7 == nj)
; CHECK-NEXT:                                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                          }
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                  if (ni >= 96 * c2 + 4 * c4 + 4) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj >= 2048 * c0 + 2) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj >= 2048 * c0 + 3) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj >= 2048 * c0 + 4) {
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                          if (nj >= 2048 * c0 + 5) {
; CHECK-NEXT:                                            Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                            if (nj >= 2048 * c0 + 6) {
; CHECK-NEXT:                                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, 2048 * c0 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                              if (2048 * c0 + 7 == nj)
; CHECK-NEXT:                                                Stmt_for_body6(96 * c2 + 4 * c4 + 3, nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                            }
; CHECK-NEXT:                                          }
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                              }
; CHECK-NEXT:                            }
; CHECK-NEXT:                          }
; CHECK-NEXT:                      } else if (96 * c2 + 3 >= ni) {
; CHECK-NEXT:                        for (int c3 = 0; c3 <= min(255, -256 * c0 + (nj - 1) / 8); c3 += 1)
; CHECK-NEXT:                          for (int c5 = 0; c5 <= min(255, nk - 256 * c1 - 1); c5 += 1) {
; CHECK-NEXT:                            // Loop Vectorizer Disabled
; CHECK-NEXT:                            // Register tiling - Points
; CHECK-NEXT:                            {
; CHECK-NEXT:                              Stmt_for_body6(96 * c2, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                              if (nj >= 2048 * c0 + 8 * c3 + 2) {
; CHECK-NEXT:                                Stmt_for_body6(96 * c2, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                if (nj >= 2048 * c0 + 8 * c3 + 3) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj >= 2048 * c0 + 8 * c3 + 4) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj >= 2048 * c0 + 8 * c3 + 5) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj >= 2048 * c0 + 8 * c3 + 6) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj >= 2048 * c0 + 8 * c3 + 7) {
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                          if (nj >= 2048 * c0 + 8 * c3 + 8)
; CHECK-NEXT:                                            Stmt_for_body6(96 * c2, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                              }
; CHECK-NEXT:                              if (ni >= 96 * c2 + 2) {
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 1, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                                if (nj >= 2048 * c0 + 8 * c3 + 2) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 1, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj >= 2048 * c0 + 8 * c3 + 3) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 1, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj >= 2048 * c0 + 8 * c3 + 4) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 1, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj >= 2048 * c0 + 8 * c3 + 5) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 1, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj >= 2048 * c0 + 8 * c3 + 6) {
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 1, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                          if (nj >= 2048 * c0 + 8 * c3 + 7) {
; CHECK-NEXT:                                            Stmt_for_body6(96 * c2 + 1, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                            if (nj >= 2048 * c0 + 8 * c3 + 8)
; CHECK-NEXT:                                              Stmt_for_body6(96 * c2 + 1, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                          }
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                                if (96 * c2 + 3 == ni) {
; CHECK-NEXT:                                  Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj >= 2048 * c0 + 8 * c3 + 2) {
; CHECK-NEXT:                                    Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj >= 2048 * c0 + 8 * c3 + 3) {
; CHECK-NEXT:                                      Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj >= 2048 * c0 + 8 * c3 + 4) {
; CHECK-NEXT:                                        Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj >= 2048 * c0 + 8 * c3 + 5) {
; CHECK-NEXT:                                          Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                                          if (nj >= 2048 * c0 + 8 * c3 + 6) {
; CHECK-NEXT:                                            Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                                            if (nj >= 2048 * c0 + 8 * c3 + 7) {
; CHECK-NEXT:                                              Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                                              if (nj >= 2048 * c0 + 8 * c3 + 8)
; CHECK-NEXT:                                                Stmt_for_body6(ni - 1, 2048 * c0 + 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                                            }
; CHECK-NEXT:                                          }
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                              }
; CHECK-NEXT:                            }
; CHECK-NEXT:                          }
; CHECK-NEXT:                      } else {
; CHECK-NEXT:                        for (int c4 = 0; c4 <= min(23, -24 * c2 + (ni - 1) / 4); c4 += 1)
; CHECK-NEXT:                          for (int c5 = 0; c5 <= min(255, nk - 256 * c1 - 1); c5 += 1) {
; CHECK-NEXT:                            // Loop Vectorizer Disabled
; CHECK-NEXT:                            // Register tiling - Points
; CHECK-NEXT:                            {
; CHECK-NEXT:                              Stmt_for_body6(96 * c2 + 4 * c4, -((nj - 1) % 8) + nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                              if (nj % 8 >= 2) {
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4 * c4, -((nj - 1) % 8) + nj, 256 * c1 + c5);
; CHECK-NEXT:                                if (nj % 8 >= 3) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 4 * c4, -((nj - 1) % 8) + nj + 1, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj % 8 >= 4) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 4 * c4, -((nj - 1) % 8) + nj + 2, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj % 8 >= 5) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 4 * c4, -((nj - 1) % 8) + nj + 3, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj % 8 >= 6) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 4 * c4, -((nj - 1) % 8) + nj + 4, 256 * c1 + c5);
; CHECK-NEXT:                                        if ((nj + 1) % 8 == 0)
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 4 * c4, nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                              }
; CHECK-NEXT:                              if (ni >= 96 * c2 + 4 * c4 + 2) {
; CHECK-NEXT:                                Stmt_for_body6(96 * c2 + 4 * c4 + 1, -((nj - 1) % 8) + nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                if (nj % 8 >= 2) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 4 * c4 + 1, -((nj - 1) % 8) + nj, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj % 8 >= 3) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 4 * c4 + 1, -((nj - 1) % 8) + nj + 1, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj % 8 >= 4) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 4 * c4 + 1, -((nj - 1) % 8) + nj + 2, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj % 8 >= 5) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 4 * c4 + 1, -((nj - 1) % 8) + nj + 3, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj % 8 >= 6) {
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 4 * c4 + 1, -((nj - 1) % 8) + nj + 4, 256 * c1 + c5);
; CHECK-NEXT:                                          if ((nj + 1) % 8 == 0)
; CHECK-NEXT:                                            Stmt_for_body6(96 * c2 + 4 * c4 + 1, nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                                if (ni >= 96 * c2 + 4 * c4 + 3) {
; CHECK-NEXT:                                  Stmt_for_body6(96 * c2 + 4 * c4 + 2, -((nj - 1) % 8) + nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                  if (nj % 8 >= 2) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 4 * c4 + 2, -((nj - 1) % 8) + nj, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj % 8 >= 3) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 4 * c4 + 2, -((nj - 1) % 8) + nj + 1, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj % 8 >= 4) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 4 * c4 + 2, -((nj - 1) % 8) + nj + 2, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj % 8 >= 5) {
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 4 * c4 + 2, -((nj - 1) % 8) + nj + 3, 256 * c1 + c5);
; CHECK-NEXT:                                          if (nj % 8 >= 6) {
; CHECK-NEXT:                                            Stmt_for_body6(96 * c2 + 4 * c4 + 2, -((nj - 1) % 8) + nj + 4, 256 * c1 + c5);
; CHECK-NEXT:                                            if ((nj + 1) % 8 == 0)
; CHECK-NEXT:                                              Stmt_for_body6(96 * c2 + 4 * c4 + 2, nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                          }
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                  if (ni >= 96 * c2 + 4 * c4 + 4) {
; CHECK-NEXT:                                    Stmt_for_body6(96 * c2 + 4 * c4 + 3, -((nj - 1) % 8) + nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                    if (nj % 8 >= 2) {
; CHECK-NEXT:                                      Stmt_for_body6(96 * c2 + 4 * c4 + 3, -((nj - 1) % 8) + nj, 256 * c1 + c5);
; CHECK-NEXT:                                      if (nj % 8 >= 3) {
; CHECK-NEXT:                                        Stmt_for_body6(96 * c2 + 4 * c4 + 3, -((nj - 1) % 8) + nj + 1, 256 * c1 + c5);
; CHECK-NEXT:                                        if (nj % 8 >= 4) {
; CHECK-NEXT:                                          Stmt_for_body6(96 * c2 + 4 * c4 + 3, -((nj - 1) % 8) + nj + 2, 256 * c1 + c5);
; CHECK-NEXT:                                          if (nj % 8 >= 5) {
; CHECK-NEXT:                                            Stmt_for_body6(96 * c2 + 4 * c4 + 3, -((nj - 1) % 8) + nj + 3, 256 * c1 + c5);
; CHECK-NEXT:                                            if (nj % 8 >= 6) {
; CHECK-NEXT:                                              Stmt_for_body6(96 * c2 + 4 * c4 + 3, -((nj - 1) % 8) + nj + 4, 256 * c1 + c5);
; CHECK-NEXT:                                              if ((nj + 1) % 8 == 0)
; CHECK-NEXT:                                                Stmt_for_body6(96 * c2 + 4 * c4 + 3, nj - 1, 256 * c1 + c5);
; CHECK-NEXT:                                            }
; CHECK-NEXT:                                          }
; CHECK-NEXT:                                        }
; CHECK-NEXT:                                      }
; CHECK-NEXT:                                    }
; CHECK-NEXT:                                  }
; CHECK-NEXT:                                }
; CHECK-NEXT:                              }
; CHECK-NEXT:                            }
; CHECK-NEXT:                          }
; CHECK-NEXT:                      }
; CHECK-NEXT:                    }
; CHECK-NEXT:                  }
; CHECK-NEXT:                }
; CHECK-NEXT:              }
; CHECK-NEXT:          }
;

; AUTO-VECTORIZATION:  fmul <4 x double>
; AUTO-VECTORIZATION:  fadd <4 x double>

; AUTO-VECTORIZATION: 36 SLP              - Number of vector instructions generated
; AUTO-VECTORIZATION:  453 licm             - Number of instructions hoisted out of loop
; AUTO-VECTORIZATION:    2 licm             - Number of load insts hoisted or sunk
; AUTO-VECTORIZATION:   32 licm             - Number of memory locations promoted to registers
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, double %alpha, double %beta, [1024 x double]* %C, [1024 x double]* %A, [1024 x double]* %B) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp39 = icmp sgt i32 %ni, 0
  br i1 %cmp39, label %for.cond1.preheader.lr.ph, label %for.end22

for.cond1.preheader.lr.ph:                        ; preds = %entry.split
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc20, %for.cond1.preheader.lr.ph
  %indvars.iv45 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next46, %for.inc20 ]
  %cmp237 = icmp sgt i32 %nj, 0
  br i1 %cmp237, label %for.cond4.preheader.lr.ph, label %for.inc20

for.cond4.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc17, %for.cond4.preheader.lr.ph
  %indvars.iv41 = phi i64 [ 0, %for.cond4.preheader.lr.ph ], [ %indvars.iv.next42, %for.inc17 ]
  %cmp535 = icmp sgt i32 %nk, 0
  br i1 %cmp535, label %for.body6.lr.ph, label %for.inc17

for.body6.lr.ph:                                  ; preds = %for.cond4.preheader
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body6.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body6.lr.ph ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv45, i64 %indvars.iv
  %tmp = load double, double* %arrayidx8, align 8
  %arrayidx12 = getelementptr inbounds [1024 x double], [1024 x double]* %B, i64 %indvars.iv, i64 %indvars.iv41
  %tmp1 = load double, double* %arrayidx12, align 8
  %mul = fmul double %tmp, %tmp1
  %arrayidx16 = getelementptr inbounds [1024 x double], [1024 x double]* %C, i64 %indvars.iv45, i64 %indvars.iv41
  %tmp2 = load double, double* %arrayidx16, align 8
  %add = fadd double %tmp2, %mul
  store double %add, double* %arrayidx16, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %wide.trip.count = zext i32 %nk to i64
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body6, label %for.cond4.for.inc17_crit_edge

for.cond4.for.inc17_crit_edge:                    ; preds = %for.body6
  br label %for.inc17

for.inc17:                                        ; preds = %for.cond4.for.inc17_crit_edge, %for.cond4.preheader
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %wide.trip.count43 = zext i32 %nj to i64
  %exitcond44 = icmp ne i64 %indvars.iv.next42, %wide.trip.count43
  br i1 %exitcond44, label %for.cond4.preheader, label %for.cond1.for.inc20_crit_edge

for.cond1.for.inc20_crit_edge:                    ; preds = %for.inc17
  br label %for.inc20

for.inc20:                                        ; preds = %for.cond1.for.inc20_crit_edge, %for.cond1.preheader
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %wide.trip.count47 = zext i32 %ni to i64
  %exitcond48 = icmp ne i64 %indvars.iv.next46, %wide.trip.count47
  br i1 %exitcond48, label %for.cond1.preheader, label %for.cond.for.end22_crit_edge

for.cond.for.end22_crit_edge:                     ; preds = %for.inc20
  br label %for.end22

for.end22:                                        ; preds = %for.cond.for.end22_crit_edge, %entry.split
  ret void
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+cx16,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt" }
