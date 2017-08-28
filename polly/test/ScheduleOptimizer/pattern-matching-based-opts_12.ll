; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -analyze -polly-ast -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 < %s \
; RUN: | FileCheck %s
;
; Test whether isolation works as expected.
;
; CHECK:    // Inter iteration alias-free
; CHECK-NEXT:    // 1st level tiling - Tiles
; CHECK-NEXT:    for (int c1 = 0; c1 <= 1; c1 += 1) {
; CHECK-NEXT:      for (int c3 = 0; c3 <= 1019; c3 += 1)
; CHECK-NEXT:        for (int c4 = 512 * c1; c4 <= min(1019, 512 * c1 + 511); c4 += 1)
; CHECK-NEXT:          CopyStmt_0(0, c3, c4);
; CHECK-NEXT:      for (int c2 = 0; c2 <= 2; c2 += 1) {
; CHECK-NEXT:        for (int c3 = 384 * c2; c3 <= min(1019, 384 * c2 + 383); c3 += 1)
; CHECK-NEXT:          for (int c5 = 512 * c1; c5 <= min(1019, 512 * c1 + 511); c5 += 1)
; CHECK-NEXT:            CopyStmt_1(c3, 0, c5);
; CHECK-NEXT:        // 1st level tiling - Points
; CHECK-NEXT:        // Register tiling - Tiles
; CHECK-NEXT:        {
; CHECK-NEXT:          for (int c3 = 0; c3 <= 30; c3 += 1) {
; CHECK-NEXT:            for (int c4 = 0; c4 <= min(47, -48 * c2 + 126); c4 += 1)
; CHECK-NEXT:              for (int c5 = 0; c5 <= min(511, -512 * c1 + 1019); c5 += 1) {
; CHECK-NEXT:                // Loop Vectorizer Disabled
; CHECK-NEXT:                // Register tiling - Points
; CHECK-NEXT:                {
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 1, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 2, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 4, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 5, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 6, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 7, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 8, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 9, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 10, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 11, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 12, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 13, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 14, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 15, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 16, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 17, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 18, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 19, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 20, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 21, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 22, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 23, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 24, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 25, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 26, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 27, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 28, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 29, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 30, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4, 32 * c3 + 31, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 1, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 2, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 4, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 5, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 6, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 7, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 8, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 9, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 10, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 11, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 12, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 13, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 14, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 15, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 16, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 17, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 18, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 19, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 20, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 21, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 22, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 23, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 24, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 25, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 26, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 27, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 28, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 29, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 30, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 1, 32 * c3 + 31, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 1, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 2, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 4, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 5, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 6, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 7, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 8, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 9, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 10, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 11, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 12, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 13, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 14, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 15, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 16, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 17, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 18, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 19, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 20, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 21, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 22, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 23, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 24, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 25, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 26, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 27, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 28, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 29, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 30, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 2, 32 * c3 + 31, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 1, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 2, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 4, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 5, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 6, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 7, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 8, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 9, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 10, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 11, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 12, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 13, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 14, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 15, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 16, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 17, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 18, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 19, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 20, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 21, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 22, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 23, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 24, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 25, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 26, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 27, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 28, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 29, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 30, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 3, 32 * c3 + 31, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 1, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 2, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 4, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 5, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 6, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 7, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 8, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 9, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 10, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 11, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 12, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 13, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 14, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 15, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 16, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 17, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 18, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 19, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 20, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 21, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 22, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 23, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 24, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 25, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 26, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 27, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 28, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 29, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 30, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 4, 32 * c3 + 31, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 1, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 2, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 4, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 5, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 6, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 7, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 8, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 9, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 10, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 11, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 12, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 13, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 14, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 15, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 16, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 17, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 18, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 19, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 20, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 21, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 22, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 23, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 24, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 25, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 26, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 27, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 28, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 29, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 30, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 5, 32 * c3 + 31, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 1, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 2, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 4, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 5, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 6, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 7, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 8, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 9, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 10, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 11, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 12, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 13, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 14, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 15, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 16, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 17, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 18, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 19, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 20, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 21, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 22, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 23, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 24, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 25, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 26, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 27, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 28, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 29, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 30, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 6, 32 * c3 + 31, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 1, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 2, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 3, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 4, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 5, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 6, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 7, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 8, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 9, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 10, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 11, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 12, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 13, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 14, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 15, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 16, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 17, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 18, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 19, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 20, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 21, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 22, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 23, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 24, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 25, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 26, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 27, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 28, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 29, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 30, 512 * c1 + c5);
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + 7, 32 * c3 + 31, 512 * c1 + c5);
; CHECK-NEXT:                }
; CHECK-NEXT:              }
; CHECK-NEXT:            if (c2 == 2)
; CHECK-NEXT:              for (int c5 = 0; c5 <= min(511, -512 * c1 + 1019); c5 += 1) {
; CHECK-NEXT:                // Loop Vectorizer Disabled
; CHECK-NEXT:                // Register tiling - Points
; CHECK-NEXT:                for (int c6 = 0; c6 <= 3; c6 += 1)
; CHECK-NEXT:                  for (int c7 = 0; c7 <= 31; c7 += 1)
; CHECK-NEXT:                    Stmt_for_body6(c6 + 1016, 32 * c3 + c7, 512 * c1 + c5);
; CHECK-NEXT:              }
; CHECK-NEXT:          }
; CHECK-NEXT:          for (int c4 = 0; c4 <= min(47, -48 * c2 + 127); c4 += 1)
; CHECK-NEXT:            for (int c5 = 0; c5 <= min(511, -512 * c1 + 1019); c5 += 1) {
; CHECK-NEXT:              // Loop Vectorizer Disabled
; CHECK-NEXT:              // Register tiling - Points
; CHECK-NEXT:              for (int c6 = 0; c6 <= min(7, -384 * c2 - 8 * c4 + 1019); c6 += 1)
; CHECK-NEXT:                for (int c7 = 0; c7 <= 27; c7 += 1)
; CHECK-NEXT:                  Stmt_for_body6(384 * c2 + 8 * c4 + c6, c7 + 992, 512 * c1 + c5);
; CHECK-NEXT:            }
; CHECK-NEXT:        }
; CHECK-NEXT:      }
; CHECK-NEXT:    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, i8 signext %alpha, i8 signext %beta, [1020 x i8]* %C, [1020 x i8]* %A, [1020 x i8]* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.inc23, %entry.split
  %indvars.iv45 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next46, %for.inc23 ]
  br label %for.body3

for.body3:                                        ; preds = %for.inc20, %for.body
  %indvars.iv42 = phi i64 [ 0, %for.body ], [ %indvars.iv.next43, %for.inc20 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body3 ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1020 x i8], [1020 x i8]* %A, i64 %indvars.iv45, i64 %indvars.iv
  %tmp = load i8, i8* %arrayidx8, align 1
  %arrayidx12 = getelementptr inbounds [1020 x i8], [1020 x i8]* %B, i64 %indvars.iv, i64 %indvars.iv42
  %tmp1 = load i8, i8* %arrayidx12, align 1
  %mul = mul i8 %tmp1, %tmp
  %arrayidx17 = getelementptr inbounds [1020 x i8], [1020 x i8]* %C, i64 %indvars.iv45, i64 %indvars.iv42
  %tmp2 = load i8, i8* %arrayidx17, align 1
  %add = add i8 %mul, %tmp2
  store i8 %add, i8* %arrayidx17, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1020
  br i1 %exitcond, label %for.body6, label %for.inc20

for.inc20:                                        ; preds = %for.body6
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond44 = icmp ne i64 %indvars.iv.next43, 1020
  br i1 %exitcond44, label %for.body3, label %for.inc23

for.inc23:                                        ; preds = %for.inc20
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond47 = icmp ne i64 %indvars.iv.next46, 1020
  br i1 %exitcond47, label %for.body, label %for.end25

for.end25:                                        ; preds = %for.inc23
  ret void
}
