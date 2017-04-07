; RUN: opt %loadPolly -analyze -polly-process-unprofitable  -polly-remarks-minimal \
; RUN:     -polly-opt-isl  -polly-pattern-matching-based-opts=true \
; RUN:     -polly-target-throughput-vector-fma=1 \
; RUN:     -polly-target-latency-vector-fma=1 \
; RUN:     -polly-ast -polly-target-vector-register-bitwidth=4096 \
; RUN:     -polly-target-1st-cache-level-associativity=3 < %s | FileCheck %s
;
;     /* Test that Polly does not crash due to configurations that can lead to
;    incorrect tile size computations.
;    The parameters are setup such that Car in `getMacroKernelParams`
;    is evaluated to 0. */
;
;    static const int N = 3000;
;
;    void f(int A[N][N], int B[N][N], int C[N][N]) {
;      for (int i = 0; i < N; i++) {
;        for (int j = 0; j < N; j++) {
;          A[i][j] = 0;
;          for (int k = 0; k < N; k++) {
;            A[i][j] += B[i][k] * C[k][j];
;          }
;        }
;      }
;    }
;
; CHECK:           // 1st level tiling - Tiles
; CHECK-NEXT:      for (int c0 = 0; c0 <= 93; c0 += 1)
; CHECK-NEXT:        for (int c1 = 0; c1 <= 93; c1 += 1) {
; CHECK-NEXT:          // 1st level tiling - Points
; CHECK-NEXT:          for (int c2 = 0; c2 <= min(31, -32 * c0 + 2999); c2 += 1)
; CHECK-NEXT:            for (int c3 = 0; c3 <= min(31, -32 * c1 + 2999); c3 += 1)
; CHECK-NEXT:              Stmt_for_body3(32 * c0 + c2, 32 * c1 + c3);
; CHECK-NEXT:        }
; CHECK-NEXT:      // Inter iteration alias-free
; CHECK-NEXT:      // Register tiling - Tiles
; CHECK-NEXT:      for (int c0 = 0; c0 <= 23; c0 += 1)
; CHECK-NEXT:        for (int c1 = 0; c1 <= 2999; c1 += 1)
; CHECK-NEXT:          for (int c2 = 0; c2 <= 2999; c2 += 1) {
; CHECK-NEXT:            // Register tiling - Points
; CHECK-NEXT:            {
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 1, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 2, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 3, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 4, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 5, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 6, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 7, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 8, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 9, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 10, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 11, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 12, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 13, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 14, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 15, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 16, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 17, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 18, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 19, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 20, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 21, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 22, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 23, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 24, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 25, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 26, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 27, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 28, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 29, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 30, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 31, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 32, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 33, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 34, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 35, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 36, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 37, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 38, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 39, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 40, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 41, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 42, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 43, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 44, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 45, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 46, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 47, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 48, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 49, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 50, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 51, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 52, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 53, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 54, c2);
; CHECK-NEXT:              Stmt_for_body8(c1, 128 * c0 + 55, c2);
; CHECK-NEXT:              if (c0 <= 22) {
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 56, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 57, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 58, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 59, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 60, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 61, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 62, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 63, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 64, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 65, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 66, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 67, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 68, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 69, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 70, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 71, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 72, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 73, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 74, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 75, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 76, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 77, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 78, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 79, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 80, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 81, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 82, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 83, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 84, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 85, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 86, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 87, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 88, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 89, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 90, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 91, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 92, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 93, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 94, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 95, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 96, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 97, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 98, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 99, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 100, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 101, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 102, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 103, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 104, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 105, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 106, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 107, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 108, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 109, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 110, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 111, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 112, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 113, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 114, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 115, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 116, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 117, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 118, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 119, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 120, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 121, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 122, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 123, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 124, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 125, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 126, c2);
; CHECK-NEXT:                Stmt_for_body8(c1, 128 * c0 + 127, c2);
; CHECK-NEXT:              }
; CHECK-NEXT:            }
; CHECK-NEXT:          }
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f([3000 x i32]* %A, [3000 x i32]* %B, [3000 x i32]* %C) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc24, %entry
  %indvars.iv4 = phi i64 [ %indvars.iv.next5, %for.inc24 ], [ 0, %entry ]
  %exitcond6 = icmp ne i64 %indvars.iv4, 3000
  br i1 %exitcond6, label %for.body, label %for.end26

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc21, %for.body
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc21 ], [ 0, %for.body ]
  %exitcond3 = icmp ne i64 %indvars.iv1, 3000
  br i1 %exitcond3, label %for.body3, label %for.end23

for.body3:                                        ; preds = %for.cond1
  %arrayidx5 = getelementptr inbounds [3000 x i32], [3000 x i32]* %A, i64 %indvars.iv4, i64 %indvars.iv1
  store i32 0, i32* %arrayidx5, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body3
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body3 ]
  %exitcond = icmp ne i64 %indvars.iv, 3000
  br i1 %exitcond, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %arrayidx12 = getelementptr inbounds [3000 x i32], [3000 x i32]* %B, i64 %indvars.iv4, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx12, align 4
  %arrayidx16 = getelementptr inbounds [3000 x i32], [3000 x i32]* %C, i64 %indvars.iv, i64 %indvars.iv1
  %tmp7 = load i32, i32* %arrayidx16, align 4
  %mul = mul nsw i32 %tmp, %tmp7
  %arrayidx20 = getelementptr inbounds [3000 x i32], [3000 x i32]* %A, i64 %indvars.iv4, i64 %indvars.iv1
  %tmp8 = load i32, i32* %arrayidx20, align 4
  %add = add nsw i32 %tmp8, %mul
  store i32 %add, i32* %arrayidx20, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond6

for.end:                                          ; preds = %for.cond6
  br label %for.inc21

for.inc21:                                        ; preds = %for.end
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond1

for.end23:                                        ; preds = %for.cond1
  br label %for.inc24

for.inc24:                                        ; preds = %for.end23
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  br label %for.cond

for.end26:                                        ; preds = %for.cond
  ret void
}
