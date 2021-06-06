; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -analyze -polly-ast -polly-target-1st-cache-level-size=0 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: < %s 2>&1 | FileCheck %s

; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -analyze -polly-ast -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 < %s 2>&1 \
; RUN: | FileCheck %s --check-prefix=EXTRACTION-OF-MACRO-KERNEL
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
; CHECK-LABEL: Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: 'bb8 => bb32' in function 'kernel_gemm':
; CHECK:    {
; CHECK-NEXT:      // 1st level tiling - Tiles
; CHECK-NEXT:      for (int c0 = 0; c0 <= 32; c0 += 1)
; CHECK-NEXT:        for (int c1 = 0; c1 <= 32; c1 += 1) {
; CHECK-NEXT:          // 1st level tiling - Points
; CHECK-NEXT:          for (int c2 = 0; c2 <= 31; c2 += 1)
; CHECK-NEXT:            for (int c3 = 0; c3 <= 31; c3 += 1)
; CHECK-NEXT:              Stmt_bb9(32 * c0 + c2, 32 * c1 + c3);
; CHECK-NEXT:        }
; CHECK-NEXT:      // Inter iteration alias-free
; CHECK-NEXT:      // Register tiling - Tiles
; CHECK-NEXT:      for (int c0 = 0; c0 <= 131; c0 += 1)
; CHECK-NEXT:        for (int c1 = 0; c1 <= 263; c1 += 1)
; CHECK-NEXT:          for (int c2 = 0; c2 <= 1023; c2 += 1) {
; CHECK-NEXT:            // Register tiling - Points
; CHECK-NEXT:            {
; CHECK-NEXT:              Stmt_Copy_0(4 * c1, 8 * c0, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1, 8 * c0 + 1, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1, 8 * c0 + 2, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1, 8 * c0 + 3, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1, 8 * c0 + 4, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1, 8 * c0 + 5, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1, 8 * c0 + 6, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1, 8 * c0 + 7, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 1, 8 * c0, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 1, 8 * c0 + 1, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 1, 8 * c0 + 2, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 1, 8 * c0 + 3, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 1, 8 * c0 + 4, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 1, 8 * c0 + 5, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 1, 8 * c0 + 6, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 1, 8 * c0 + 7, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 2, 8 * c0, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 2, 8 * c0 + 1, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 2, 8 * c0 + 2, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 2, 8 * c0 + 3, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 2, 8 * c0 + 4, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 2, 8 * c0 + 5, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 2, 8 * c0 + 6, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 2, 8 * c0 + 7, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 3, 8 * c0, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 3, 8 * c0 + 1, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 3, 8 * c0 + 2, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 3, 8 * c0 + 3, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 3, 8 * c0 + 4, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 3, 8 * c0 + 5, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 3, 8 * c0 + 6, c2);
; CHECK-NEXT:              Stmt_Copy_0(4 * c1 + 3, 8 * c0 + 7, c2);
; CHECK-NEXT:            }
; CHECK-NEXT:          }
; CHECK-NEXT:    }
;
; EXTRACTION-OF-MACRO-KERNEL-LABEL: Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: 'bb8 => bb32' in function 'kernel_gemm':
; EXTRACTION-OF-MACRO-KERNEL:    {
; EXTRACTION-OF-MACRO-KERNEL-NEXT:      // 1st level tiling - Tiles
; EXTRACTION-OF-MACRO-KERNEL-NEXT:      for (int c0 = 0; c0 <= 32; c0 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:        for (int c1 = 0; c1 <= 32; c1 += 1) {
; EXTRACTION-OF-MACRO-KERNEL-NEXT:          // 1st level tiling - Points
; EXTRACTION-OF-MACRO-KERNEL-NEXT:          for (int c2 = 0; c2 <= 31; c2 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:            for (int c3 = 0; c3 <= 31; c3 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:              Stmt_bb9(32 * c0 + c2, 32 * c1 + c3);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:        }
; EXTRACTION-OF-MACRO-KERNEL-NEXT:      // Inter iteration alias-free
; EXTRACTION-OF-MACRO-KERNEL-NEXT:      // 1st level tiling - Tiles
; EXTRACTION-OF-MACRO-KERNEL-NEXT:      for (int c1 = 0; c1 <= 3; c1 += 1) {
; EXTRACTION-OF-MACRO-KERNEL-NEXT:        for (int c3 = 0; c3 <= 1055; c3 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:          for (int c4 = 256 * c1; c4 <= 256 * c1 + 255; c4 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:            CopyStmt_0(0, c3, c4);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:        for (int c2 = 0; c2 <= 10; c2 += 1) {
; EXTRACTION-OF-MACRO-KERNEL-NEXT:          for (int c6 = 96 * c2; c6 <= 96 * c2 + 95; c6 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:            for (int c7 = 256 * c1; c7 <= 256 * c1 + 255; c7 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:              CopyStmt_1(0, c1, c2, c6, c7);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:          // 1st level tiling - Points
; EXTRACTION-OF-MACRO-KERNEL-NEXT:          // Register tiling - Tiles
; EXTRACTION-OF-MACRO-KERNEL-NEXT:          for (int c3 = 0; c3 <= 131; c3 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:            for (int c4 = 0; c4 <= 23; c4 += 1)
; EXTRACTION-OF-MACRO-KERNEL-NEXT:              for (int c5 = 0; c5 <= 255; c5 += 1) {
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                // Loop Vectorizer Disabled
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                // Register tiling - Points
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                {
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 1, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 2, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 5, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 6, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 7, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 1, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 2, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 5, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 6, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 7, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 1, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 2, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 5, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 6, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 7, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 1, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 2, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 5, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 6, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 7, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL-NEXT:                }
; EXTRACTION-OF-MACRO-KERNEL-NEXT:              }
; EXTRACTION-OF-MACRO-KERNEL-NEXT:        }
; EXTRACTION-OF-MACRO-KERNEL-NEXT:      }
; EXTRACTION-OF-MACRO-KERNEL-NEXT:    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %arg, i32 %arg1, i32 %arg2, double %arg3, double %arg4, [1056 x double]* %arg5, [1024 x double]* %arg6, [1056 x double]* %arg7) #0 {
bb:
  br label %bb8

bb8:                                              ; preds = %bb29, %bb
  %tmp = phi i64 [ 0, %bb ], [ %tmp30, %bb29 ]
  br label %bb9

bb9:                                              ; preds = %bb26, %bb8
  %tmp10 = phi i64 [ 0, %bb8 ], [ %tmp27, %bb26 ]
  %tmp11 = getelementptr inbounds [1056 x double], [1056 x double]* %arg5, i64 %tmp, i64 %tmp10
  %tmp12 = load double, double* %tmp11, align 8
  %tmp13 = fmul double %tmp12, %arg4
  store double %tmp13, double* %tmp11, align 8
  br label %Copy_0

Copy_0:                                             ; preds = %Copy_0, %bb9
  %tmp15 = phi i64 [ 0, %bb9 ], [ %tmp24, %Copy_0 ]
  %tmp16 = getelementptr inbounds [1024 x double], [1024 x double]* %arg6, i64 %tmp, i64 %tmp15
  %tmp17 = load double, double* %tmp16, align 8
  %tmp18 = fmul double %tmp17, %arg3
  %tmp19 = getelementptr inbounds [1056 x double], [1056 x double]* %arg7, i64 %tmp15, i64 %tmp10
  %tmp20 = load double, double* %tmp19, align 8
  %tmp21 = fmul double %tmp18, %tmp20
  %tmp22 = load double, double* %tmp11, align 8
  %tmp23 = fadd double %tmp22, %tmp21
  store double %tmp23, double* %tmp11, align 8
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
