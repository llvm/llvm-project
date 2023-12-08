; RUN: opt %loadPolly -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=2 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=128 \
; RUN: -polly-target-2nd-cache-level-size=262144 \
; RUN: -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s
;
; Test whether isolation works as expected.
;
; CHECK:               // 1st level tiling - Tiles
; CHECK-NEXT:          for (int c0 = 0; c0 <= 1; c0 += 1)
; CHECK-NEXT:            for (int c1 = 0; c1 <= 6; c1 += 1) {
; CHECK-NEXT:              for (int c3 = 1536 * c0; c3 <= min(1999, 1536 * c0 + 1535); c3 += 1)
; CHECK-NEXT:                for (int c4 = 307 * c1; c4 <= min(1999, 307 * c1 + 306); c4 += 1)
; CHECK-NEXT:                  CopyStmt_0(0, c3, c4);
; CHECK-NEXT:              for (int c2 = 0; c2 <= 24; c2 += 1) {
; CHECK-NEXT:                  for (int c6 = 80 * c2; c6 <= 80 * c2 + 79; c6 += 1)
; CHECK-NEXT:                    for (int c7 = 307 * c1; c7 <= min(1999, 307 * c1 + 306); c7 += 1)
; CHECK-NEXT:                      CopyStmt_1(c0, c1, c2, c6, c7);
; CHECK-NEXT:                // 1st level tiling - Points
; CHECK-NEXT:                // Register tiling - Tiles
; CHECK-NEXT:                {
; CHECK-NEXT:                  for (int c3 = 0; c3 <= min(255, -256 * c0 + 332); c3 += 1)
; CHECK-NEXT:                    for (int c4 = 0; c4 <= 15; c4 += 1)
; CHECK-NEXT:                      for (int c5 = 0; c5 <= min(306, -307 * c1 + 1999); c5 += 1) {
; CHECK-NEXT:                        // Loop Vectorizer Disabled
; CHECK-NEXT:                        // Register tiling - Points
; CHECK-NEXT:                        {
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4, 1536 * c0 + 6 * c3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4, 1536 * c0 + 6 * c3 + 1, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4, 1536 * c0 + 6 * c3 + 2, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4, 1536 * c0 + 6 * c3 + 3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4, 1536 * c0 + 6 * c3 + 4, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4, 1536 * c0 + 6 * c3 + 5, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 1, 1536 * c0 + 6 * c3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 1, 1536 * c0 + 6 * c3 + 1, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 1, 1536 * c0 + 6 * c3 + 2, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 1, 1536 * c0 + 6 * c3 + 3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 1, 1536 * c0 + 6 * c3 + 4, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 1, 1536 * c0 + 6 * c3 + 5, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 2, 1536 * c0 + 6 * c3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 2, 1536 * c0 + 6 * c3 + 1, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 2, 1536 * c0 + 6 * c3 + 2, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 2, 1536 * c0 + 6 * c3 + 3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 2, 1536 * c0 + 6 * c3 + 4, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 2, 1536 * c0 + 6 * c3 + 5, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 3, 1536 * c0 + 6 * c3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 3, 1536 * c0 + 6 * c3 + 1, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 3, 1536 * c0 + 6 * c3 + 2, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 3, 1536 * c0 + 6 * c3 + 3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 3, 1536 * c0 + 6 * c3 + 4, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 3, 1536 * c0 + 6 * c3 + 5, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 4, 1536 * c0 + 6 * c3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 4, 1536 * c0 + 6 * c3 + 1, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 4, 1536 * c0 + 6 * c3 + 2, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 4, 1536 * c0 + 6 * c3 + 3, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 4, 1536 * c0 + 6 * c3 + 4, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 4, 1536 * c0 + 6 * c3 + 5, 307 * c1 + c5);
; CHECK-NEXT:                        }
; CHECK-NEXT:                      }
; CHECK-NEXT:                  if (c0 == 1)
; CHECK-NEXT:                    for (int c4 = 0; c4 <= 15; c4 += 1)
; CHECK-NEXT:                      for (int c5 = 0; c5 <= min(306, -307 * c1 + 1999); c5 += 1) {
; CHECK-NEXT:                        // Loop Vectorizer Disabled
; CHECK-NEXT:                        // Register tiling - Points
; CHECK-NEXT:                        {
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4, 1998, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4, 1999, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 1, 1998, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 1, 1999, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 2, 1998, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 2, 1999, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 3, 1998, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 3, 1999, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 4, 1998, 307 * c1 + c5);
; CHECK-NEXT:                          Stmt_for_body6(80 * c2 + 5 * c4 + 4, 1999, 307 * c1 + c5);
; CHECK-NEXT:                        }
; CHECK-NEXT:                      }
; CHECK-NEXT:                }
; CHECK-NEXT:              }
; CHECK-NEXT:            }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, double %alpha, double %beta, ptr %C, ptr %A, ptr %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.inc20, %entry.split
  %indvars.iv41 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next42, %for.inc20 ]
  br label %for.body3

for.body3:                                        ; preds = %for.inc17, %for.body
  %indvars.iv38 = phi i64 [ 0, %for.body ], [ %indvars.iv.next39, %for.inc17 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body3 ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [2000 x double], ptr %A, i64 %indvars.iv41, i64 %indvars.iv
  %tmp = load double, ptr %arrayidx8, align 8
  %arrayidx12 = getelementptr inbounds [2000 x double], ptr %B, i64 %indvars.iv, i64 %indvars.iv38
  %tmp1 = load double, ptr %arrayidx12, align 8
  %mul = fmul double %tmp, %tmp1
  %arrayidx16 = getelementptr inbounds [2000 x double], ptr %C, i64 %indvars.iv41, i64 %indvars.iv38
  %tmp2 = load double, ptr %arrayidx16, align 8
  %add = fadd double %tmp2, %mul
  store double %add, ptr %arrayidx16, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 2000
  br i1 %exitcond, label %for.body6, label %for.inc17

for.inc17:                                        ; preds = %for.body6
  %indvars.iv.next39 = add nuw nsw i64 %indvars.iv38, 1
  %exitcond40 = icmp ne i64 %indvars.iv.next39, 2000
  br i1 %exitcond40, label %for.body3, label %for.inc20

for.inc20:                                        ; preds = %for.inc17
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %exitcond43 = icmp ne i64 %indvars.iv.next42, 2000
  br i1 %exitcond43, label %for.body, label %for.end22

for.end22:                                        ; preds = %for.inc20
  ret void
}
