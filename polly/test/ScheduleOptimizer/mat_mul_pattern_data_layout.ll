; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-2nd-cache-level-size=262144 \
; RUN: -polly-optimized-scops \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: < %s 2>&1 | FileCheck %s
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
; CHECK:        double Packed_B[ { [] -> [(256)] } ][ { [] -> [(256)] } ][ { [] -> [(8)] } ];
; CHECK-NEXT:        double Packed_A[ { [] -> [(24)] } ][ { [] -> [(256)] } ][ { [] -> [(4)] } ]; // Element size 8
;
; CHECK:                { Stmt_Copy_0[i0, i1, i2] -> MemRef_arg6[i0, i2] };
; CHECK-NEXT:           new: { Stmt_Copy_0[i0, i1, i2] -> Packed_A[o0, o1, o2] : 256*floor((-i2 + o1)/256) = -i2 + o1 and 4*floor((-i0 + o2)/4) = -i0 + o2 and 0 <= o1 <= 255 and 0 <= o2 <= 3 and -3 + i0 - 4o0 <= 96*floor((i0)/96) <= i0 - 4o0 };
;
; CHECK:                { Stmt_Copy_0[i0, i1, i2] -> MemRef_arg7[i2, i1] };
; CHECK-NEXT:           new: { Stmt_Copy_0[i0, i1, i2] -> Packed_B[o0, o1, o2] : 256*floor((-i2 + o1)/256) = -i2 + o1 and 8*floor((-i1 + o2)/8) = -i1 + o2 and 0 <= o1 <= 255 and 0 <= o2 <= 7 and -7 + i1 - 8o0 <= 2048*floor((i1)/2048) <= i1 - 8o0 };
;
; CHECK:    	CopyStmt_0
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                { CopyStmt_0[i0, i1, i2] : 0 <= i0 <= 1055 and 0 <= i1 <= 1055 and 0 <= i2 <= 1023 };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                ;
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                null;
; CHECK-NEXT:           new: { CopyStmt_0[i0, i1, i2] -> Packed_B[o0, o1, o2] : 256*floor((-i2 + o1)/256) = -i2 + o1 and 8*floor((-i1 + o2)/8) = -i1 + o2 and 0 <= o1 <= 255 and 0 <= o2 <= 7 and -7 + i1 - 8o0 <= 2048*floor((i1)/2048) <= i1 - 8o0 };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                null;
; CHECK-NEXT:           new: { CopyStmt_0[i0, i1, i2] -> MemRef_arg7[i2, i1] };
; CHECK-NEXT:    	CopyStmt_1
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                { CopyStmt_1[i0, i1, i2] : 0 <= i0 <= 1055 and 0 <= i1 <= 1055 and 0 <= i2 <= 1023 };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                ;
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                null;
; CHECK-NEXT:           new: { CopyStmt_1[i0, i1, i2] -> Packed_A[o0, o1, o2] : 256*floor((-i2 + o1)/256) = -i2 + o1 and 4*floor((-i0 + o2)/4) = -i0 + o2 and 0 <= o1 <= 255 and 0 <= o2 <= 3 and -3 + i0 - 4o0 <= 96*floor((i0)/96) <= i0 - 4o0 };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                null;
; CHECK-NEXT:           new: { CopyStmt_1[i0, i1, i2] -> MemRef_arg6[i0, i2] };
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

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+cx16,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt" }
