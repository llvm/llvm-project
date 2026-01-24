; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl>' -polly-pattern-matching-based-opts=false -debug -polly-tc-opt -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl>' -polly-pattern-matching-based-opts=true -debug -polly-tc-opt -disable-output < %s 2>&1 | FileCheck %s --check-prefix=PATTERN-MATCHING-OPTS
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -polly-print-ast -polly-pattern-matching-based-opts=true -polly-ast-detect-parallel -disable-output < %s | FileCheck %s --check-prefix=PARALLEL-AST
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl>' -polly-pattern-matching-based-opts=true -stats -disable-output < %s 2>&1 | FileCheck %s --check-prefix=STATS -match-full-lines
; REQUIRES: asserts
;
;    /* C := alpha*A*B + beta*C */
;    for (i = 0; i < _PB_NI; i++)
;      for (j = 0; j < _PB_NJ; j++)
;        {
;	   C[i][j] *= beta;
;	   for (k = 0; k < _PB_NK; ++k)
;	     C[i][j] += alpha * A[i][k] * B[k][j];
;        }
;
; CHECK-NOT: The matrix multiplication pattern was detected
; CHECK-NOT: The tensor contraction pattern was detected
; PATTERN-MATCHING-OPTS: The tensor contraction pattern was detected
; PATTERN-MATCHING-OPTS: The matrix multiplication pattern was detected
;
; PARALLEL-AST:      // 1st level tiling - Tiles
; PARALLEL-AST-NEXT: #pragma known-parallel
; PARALLEL-AST-NEXT: for (int c0 = 0; c0 <= 32; c0 += 1)
; PARALLEL-AST-NEXT:   for (int c1 = 0; c1 <= 32; c1 += 1) {
; PARALLEL-AST-NEXT:     // 1st level tiling - Points
; PARALLEL-AST-NEXT:     for (int c2 = 0; c2 <= 31; c2 += 1)
; PARALLEL-AST-NEXT:       #pragma simd
; PARALLEL-AST-NEXT:       for (int c3 = 0; c3 <= 31; c3 += 1)
; PARALLEL-AST-NEXT:         Stmt_bb9(32 * c0 + c2, 32 * c1 + c3);
; PARALLEL-AST-NEXT:   }
; PARALLEL-AST-NEXT: // 1st level tiling - Tiles
; PARALLEL-AST-NEXT: #pragma minimal dependence distance: 1
; PARALLEL-AST-NEXT: for (int c0 = 0; c0 <= 1; c0 += 1)
; PARALLEL-AST-NEXT:   #pragma minimal dependence distance: 1
; PARALLEL-AST-NEXT:   for (int c1 = 0; c1 <= 2; c1 += 1) {
; PARALLEL-AST-NEXT:     #pragma known-parallel
; PARALLEL-AST-NEXT:     for (int c3 = 1024 * c0; c3 <= min(1055, 1024 * c0 + 1023); c3 += 1)
; PARALLEL-AST-NEXT:       #pragma simd
; PARALLEL-AST-NEXT:       for (int c4 = 384 * c1; c4 <= min(1023, 384 * c1 + 383); c4 += 1)
; PARALLEL-AST-NEXT:         CopyStmt_0(0, c3, c4);
; PARALLEL-AST-NEXT:     #pragma minimal dependence distance: 1
; PARALLEL-AST-NEXT:     for (int c2 = 0; c2 <= 16; c2 += 1) {
; PARALLEL-AST-NEXT:       #pragma known-parallel
; PARALLEL-AST-NEXT:       for (int c6 = 64 * c2; c6 <= min(1055, 64 * c2 + 63); c6 += 1)
; PARALLEL-AST-NEXT:         #pragma simd
; PARALLEL-AST-NEXT:         for (int c7 = 384 * c1; c7 <= min(1023, 384 * c1 + 383); c7 += 1)
; PARALLEL-AST-NEXT:           CopyStmt_1(c0, c1, c2, c6, c7);
; PARALLEL-AST-NEXT:       // 1st level tiling - Points
; PARALLEL-AST-NEXT:       // Register tiling - Tiles
; PARALLEL-AST-NEXT:       #pragma known-parallel
; PARALLEL-AST-NEXT:       for (int c3 = 0; c3 <= min(255, -256 * c0 + 263); c3 += 1)
; PARALLEL-AST-NEXT:         for (int c4 = 0; c4 <= min(15, -16 * c2 + 263); c4 += 1)
; PARALLEL-AST-NEXT:           #pragma minimal dependence distance: 1
; PARALLEL-AST-NEXT:           for (int c5 = 0; c5 <= min(383, -384 * c1 + 1023); c5 += 1) {
; PARALLEL-AST-NEXT:             // Loop Vectorizer Disabled
; PARALLEL-AST-NEXT:             // Register tiling - Points
; PARALLEL-AST-NEXT:             {
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4, 1024 * c0 + 4 * c3, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4, 1024 * c0 + 4 * c3 + 1, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4, 1024 * c0 + 4 * c3 + 2, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4, 1024 * c0 + 4 * c3 + 3, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 1, 1024 * c0 + 4 * c3, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 1, 1024 * c0 + 4 * c3 + 1, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 1, 1024 * c0 + 4 * c3 + 2, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 1, 1024 * c0 + 4 * c3 + 3, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 2, 1024 * c0 + 4 * c3, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 2, 1024 * c0 + 4 * c3 + 1, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 2, 1024 * c0 + 4 * c3 + 2, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 2, 1024 * c0 + 4 * c3 + 3, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 3, 1024 * c0 + 4 * c3, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 3, 1024 * c0 + 4 * c3 + 1, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 3, 1024 * c0 + 4 * c3 + 2, 384 * c1 + c5);
; PARALLEL-AST-NEXT:               Stmt_Copy_0(64 * c2 + 4 * c4 + 3, 1024 * c0 + 4 * c3 + 3, 384 * c1 + c5);
; PARALLEL-AST-NEXT:             }
; PARALLEL-AST-NEXT:           }
;
; STATS:  1 polly-opt-isl    - Number of matrix multiplication patterns detected and optimized
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %arg, i32 %arg1, i32 %arg2, double %arg3, double %arg4, ptr %arg5, ptr %arg6, ptr %arg7) #0 {
bb:
  br label %bb8

bb8:                                              ; preds = %bb29, %bb
  %tmp = phi i64 [ 0, %bb ], [ %tmp30, %bb29 ]
  br label %bb9

bb9:                                              ; preds = %bb26, %bb8
  %tmp10 = phi i64 [ 0, %bb8 ], [ %tmp27, %bb26 ]
  %tmp11 = getelementptr inbounds [1056 x double], ptr %arg5, i64 %tmp, i64 %tmp10
  %tmp12 = load double, ptr %tmp11, align 8
  %tmp13 = fmul double %tmp12, %arg4
  store double %tmp13, ptr %tmp11, align 8
  br label %Copy_0

Copy_0:                                             ; preds = %Copy_0, %bb9
  %tmp15 = phi i64 [ 0, %bb9 ], [ %tmp24, %Copy_0 ]
  %tmp16 = getelementptr inbounds [1024 x double], ptr %arg6, i64 %tmp, i64 %tmp15
  %tmp17 = load double, ptr %tmp16, align 8
  %tmp18 = fmul double %tmp17, %arg3
  %tmp19 = getelementptr inbounds [1056 x double], ptr %arg7, i64 %tmp15, i64 %tmp10
  %tmp20 = load double, ptr %tmp19, align 8
  %tmp21 = fmul double %tmp18, %tmp20
  %tmp22 = load double, ptr %tmp11, align 8
  %tmp23 = fadd double %tmp22, %tmp21
  store double %tmp23, ptr %tmp11, align 8
  %tmp24 = add nuw nsw i64 %tmp15, 1
  %tmp25 = icmp ne i64 %tmp24, 1024
  br i1 %tmp25, label %Copy_0, label %bb26

bb26:                                             ; preds = %Copy_0
  %tmp27 = add nuw nsw i64 %tmp10, 1
  %tmp28 = icmp ne i64 %tmp27, 1056
  br i1 %tmp28, label %bb9, label %bb29

bb29:                                             ; preds = %bb26
  %tmp30 = add nuw nsw i64 %tmp, 1
  %tmp31 = icmp ne i64 %tmp30, 1056
  br i1 %tmp31, label %bb8, label %bb32

bb32:                                             ; preds = %bb29
  ret void
}
