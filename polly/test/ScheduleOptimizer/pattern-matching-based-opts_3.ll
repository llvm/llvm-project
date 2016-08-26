; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true -polly-target-througput-vector-fma=1 -polly-target-latency-vector-fma=8 -analyze -polly-ast < %s 2>&1 | FileCheck %s
; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true -polly-target-througput-vector-fma=1 -polly-target-latency-vector-fma=8 -analyze -polly-ast -polly-target-cache-level-associativity=8,8 -polly-target-cache-level-sizes=32768,262144 < %s 2>&1 | FileCheck %s --check-prefix=EXTRACTION-OF-MACRO-KERNEL
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
; CHECK:    {
; CHECK:      // 1st level tiling - Tiles
; CHECK:      for (int c0 = 0; c0 <= 32; c0 += 1)
; CHECK:        for (int c1 = 0; c1 <= 32; c1 += 1) {
; CHECK:          // 1st level tiling - Points
; CHECK:          for (int c2 = 0; c2 <= 31; c2 += 1)
; CHECK:            for (int c3 = 0; c3 <= 31; c3 += 1)
; CHECK:              Stmt_bb14(32 * c0 + c2, 32 * c1 + c3);
; CHECK:        }
; CHECK:      // Register tiling - Tiles
; CHECK:      for (int c0 = 0; c0 <= 263; c0 += 1)
; CHECK:        for (int c1 = 0; c1 <= 131; c1 += 1)
; CHECK:          for (int c2 = 0; c2 <= 1023; c2 += 1) {
; CHECK:            // Register tiling - Points
; CHECK:            // 1st level tiling - Tiles
; CHECK:            // 1st level tiling - Points
; CHECK:            {
; CHECK:              Stmt_bb24(4 * c0, 8 * c1, c2);
; CHECK:              Stmt_bb24(4 * c0, 8 * c1 + 1, c2);
; CHECK:              Stmt_bb24(4 * c0, 8 * c1 + 2, c2);
; CHECK:              Stmt_bb24(4 * c0, 8 * c1 + 3, c2);
; CHECK:              Stmt_bb24(4 * c0, 8 * c1 + 4, c2);
; CHECK:              Stmt_bb24(4 * c0, 8 * c1 + 5, c2);
; CHECK:              Stmt_bb24(4 * c0, 8 * c1 + 6, c2);
; CHECK:              Stmt_bb24(4 * c0, 8 * c1 + 7, c2);
; CHECK:              Stmt_bb24(4 * c0 + 1, 8 * c1, c2);
; CHECK:              Stmt_bb24(4 * c0 + 1, 8 * c1 + 1, c2);
; CHECK:              Stmt_bb24(4 * c0 + 1, 8 * c1 + 2, c2);
; CHECK:              Stmt_bb24(4 * c0 + 1, 8 * c1 + 3, c2);
; CHECK:              Stmt_bb24(4 * c0 + 1, 8 * c1 + 4, c2);
; CHECK:              Stmt_bb24(4 * c0 + 1, 8 * c1 + 5, c2);
; CHECK:              Stmt_bb24(4 * c0 + 1, 8 * c1 + 6, c2);
; CHECK:              Stmt_bb24(4 * c0 + 1, 8 * c1 + 7, c2);
; CHECK:              Stmt_bb24(4 * c0 + 2, 8 * c1, c2);
; CHECK:              Stmt_bb24(4 * c0 + 2, 8 * c1 + 1, c2);
; CHECK:              Stmt_bb24(4 * c0 + 2, 8 * c1 + 2, c2);
; CHECK:              Stmt_bb24(4 * c0 + 2, 8 * c1 + 3, c2);
; CHECK:              Stmt_bb24(4 * c0 + 2, 8 * c1 + 4, c2);
; CHECK:              Stmt_bb24(4 * c0 + 2, 8 * c1 + 5, c2);
; CHECK:              Stmt_bb24(4 * c0 + 2, 8 * c1 + 6, c2);
; CHECK:              Stmt_bb24(4 * c0 + 2, 8 * c1 + 7, c2);
; CHECK:              Stmt_bb24(4 * c0 + 3, 8 * c1, c2);
; CHECK:              Stmt_bb24(4 * c0 + 3, 8 * c1 + 1, c2);
; CHECK:              Stmt_bb24(4 * c0 + 3, 8 * c1 + 2, c2);
; CHECK:              Stmt_bb24(4 * c0 + 3, 8 * c1 + 3, c2);
; CHECK:              Stmt_bb24(4 * c0 + 3, 8 * c1 + 4, c2);
; CHECK:              Stmt_bb24(4 * c0 + 3, 8 * c1 + 5, c2);
; CHECK:              Stmt_bb24(4 * c0 + 3, 8 * c1 + 6, c2);
; CHECK:              Stmt_bb24(4 * c0 + 3, 8 * c1 + 7, c2);
; CHECK:            }
; CHECK:          }
; CHECK:    }
;
; EXTRACTION-OF-MACRO-KERNEL:      // 1st level tiling - Tiles
; EXTRACTION-OF-MACRO-KERNEL:      for (int c0 = 0; c0 <= 65; c0 += 1)
; EXTRACTION-OF-MACRO-KERNEL:        for (int c1 = 0; c1 <= 3; c1 += 1)
; EXTRACTION-OF-MACRO-KERNEL:          for (int c2 = 0; c2 <= 10; c2 += 1) {
; EXTRACTION-OF-MACRO-KERNEL:            // 1st level tiling - Points
; EXTRACTION-OF-MACRO-KERNEL:            // Register tiling - Tiles
; EXTRACTION-OF-MACRO-KERNEL:            for (int c3 = 0; c3 <= 3; c3 += 1)
; EXTRACTION-OF-MACRO-KERNEL:              for (int c4 = 0; c4 <= 11; c4 += 1)
; EXTRACTION-OF-MACRO-KERNEL:                for (int c5 = 0; c5 <= 255; c5 += 1) {
; EXTRACTION-OF-MACRO-KERNEL:                  // Register tiling - Points
; EXTRACTION-OF-MACRO-KERNEL:                  // 1st level tiling - Tiles
; EXTRACTION-OF-MACRO-KERNEL:                  // 1st level tiling - Points
; EXTRACTION-OF-MACRO-KERNEL:                  {
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3, 96 * c2 + 8 * c4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3, 96 * c2 + 8 * c4 + 1, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3, 96 * c2 + 8 * c4 + 2, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3, 96 * c2 + 8 * c4 + 3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3, 96 * c2 + 8 * c4 + 4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3, 96 * c2 + 8 * c4 + 5, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3, 96 * c2 + 8 * c4 + 6, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3, 96 * c2 + 8 * c4 + 7, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 1, 96 * c2 + 8 * c4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 1, 96 * c2 + 8 * c4 + 1, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 1, 96 * c2 + 8 * c4 + 2, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 1, 96 * c2 + 8 * c4 + 3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 1, 96 * c2 + 8 * c4 + 4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 1, 96 * c2 + 8 * c4 + 5, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 1, 96 * c2 + 8 * c4 + 6, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 1, 96 * c2 + 8 * c4 + 7, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 2, 96 * c2 + 8 * c4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 2, 96 * c2 + 8 * c4 + 1, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 2, 96 * c2 + 8 * c4 + 2, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 2, 96 * c2 + 8 * c4 + 3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 2, 96 * c2 + 8 * c4 + 4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 2, 96 * c2 + 8 * c4 + 5, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 2, 96 * c2 + 8 * c4 + 6, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 2, 96 * c2 + 8 * c4 + 7, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 3, 96 * c2 + 8 * c4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 3, 96 * c2 + 8 * c4 + 1, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 3, 96 * c2 + 8 * c4 + 2, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 3, 96 * c2 + 8 * c4 + 3, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 3, 96 * c2 + 8 * c4 + 4, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 3, 96 * c2 + 8 * c4 + 5, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 3, 96 * c2 + 8 * c4 + 6, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                    Stmt_bb24(16 * c0 + 4 * c3 + 3, 96 * c2 + 8 * c4 + 7, 256 * c1 + c5);
; EXTRACTION-OF-MACRO-KERNEL:                  }
; EXTRACTION-OF-MACRO-KERNEL:                }
; EXTRACTION-OF-MACRO-KERNEL:          }
; EXTRACTION-OF-MACRO-KERNEL:    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %arg, i32 %arg1, i32 %arg2, double %arg3, double %arg4, [1056 x double]* %arg5, [1024 x double]* %arg6, [1056 x double]* %arg7) #0 {
bb:
  br label %bb8

bb8:                                              ; preds = %bb39, %bb
  %tmp = phi i32 [ 0, %bb ], [ %tmp40, %bb39 ]
  %tmp9 = icmp slt i32 %tmp, 1056
  br i1 %tmp9, label %bb10, label %bb41

bb10:                                             ; preds = %bb8
  br label %bb11

bb11:                                             ; preds = %bb37, %bb10
  %tmp12 = phi i32 [ 0, %bb10 ], [ %tmp38, %bb37 ]
  %tmp13 = icmp slt i32 %tmp12, 1056
  br i1 %tmp13, label %bb14, label %bb39

bb14:                                             ; preds = %bb11
  %tmp15 = sext i32 %tmp12 to i64
  %tmp16 = sext i32 %tmp to i64
  %tmp17 = getelementptr inbounds [1056 x double], [1056 x double]* %arg5, i64 %tmp16
  %tmp18 = getelementptr inbounds [1056 x double], [1056 x double]* %tmp17, i64 0, i64 %tmp15
  %tmp19 = load double, double* %tmp18, align 8
  %tmp20 = fmul double %tmp19, %arg4
  store double %tmp20, double* %tmp18, align 8
  br label %bb21

bb21:                                             ; preds = %bb24, %bb14
  %tmp22 = phi i32 [ 0, %bb14 ], [ %tmp36, %bb24 ]
  %tmp23 = icmp slt i32 %tmp22, 1024
  br i1 %tmp23, label %bb24, label %bb37

bb24:                                             ; preds = %bb21
  %tmp25 = sext i32 %tmp22 to i64
  %tmp26 = getelementptr inbounds [1024 x double], [1024 x double]* %arg6, i64 %tmp16
  %tmp27 = getelementptr inbounds [1024 x double], [1024 x double]* %tmp26, i64 0, i64 %tmp25
  %tmp28 = load double, double* %tmp27, align 8
  %tmp29 = fmul double %arg3, %tmp28
  %tmp30 = getelementptr inbounds [1056 x double], [1056 x double]* %arg7, i64 %tmp25
  %tmp31 = getelementptr inbounds [1056 x double], [1056 x double]* %tmp30, i64 0, i64 %tmp15
  %tmp32 = load double, double* %tmp31, align 8
  %tmp33 = fmul double %tmp29, %tmp32
  %tmp34 = load double, double* %tmp18, align 8
  %tmp35 = fadd double %tmp34, %tmp33
  store double %tmp35, double* %tmp18, align 8
  %tmp36 = add nsw i32 %tmp22, 1
  br label %bb21

bb37:                                             ; preds = %bb21
  %tmp38 = add nsw i32 %tmp12, 1
  br label %bb11

bb39:                                             ; preds = %bb11
  %tmp40 = add nsw i32 %tmp, 1
  br label %bb8

bb41:                                             ; preds = %bb8
  ret void
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+cx16,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt" }
