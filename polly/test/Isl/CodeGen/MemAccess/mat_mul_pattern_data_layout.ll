; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true -polly-target-througput-vector-fma=1 -polly-target-latency-vector-fma=8 -polly-target-cache-level-associativity=8,8 -polly-target-cache-level-sizes=32768,262144 -polly-codegen -S < %s 2>&1 | FileCheck %s
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
; CHECK:define internal void @kernel_gemm(i32 %arg, i32 %arg1, i32 %arg2, double %arg3, double %arg4, [1056 x double]* %arg5, [1024 x double]* %arg6, [1056 x double]* %arg7) #0 {
; CHECK:bb:
; CHECK:  %arg3.s2a = alloca double
; CHECK:  %arg4.s2a = alloca double
; CHECK:  %Packed_A = alloca [1024 x [4 x double]]
; CHECK:  %Packed_B = alloca [3072 x [8 x double]]
; CHECK:  br label %polly.split_new_and_old
;
; CHECK:polly.stmt.bb14398:                               ; preds = %polly.stmt.bb14379
; CHECK:  %arg3.s2a.reload399 = load double, double* %arg3.s2a
; CHECK:  %polly.access.cast.Packed_A400 = bitcast [1024 x [4 x double]]* %Packed_A to double*
; CHECK:  %243 = mul nsw i64 256, %polly.indvar95
; CHECK:  %244 = add nsw i64 %243, %polly.indvar107
; CHECK:  %polly.access.add.Packed_A401 = add nsw i64 0, %244
; CHECK:  %polly.access.mul.Packed_A402 = mul nsw i64 %polly.access.add.Packed_A401, 4
; CHECK:  %polly.access.add.Packed_A403 = add nsw i64 %polly.access.mul.Packed_A402, 2
; CHECK:  %polly.access.Packed_A404 = getelementptr double, double* %polly.access.cast.Packed_A400, i64 %polly.access.add.Packed_A403
; CHECK:  %tmp17_p_scalar_405 = load double, double* %polly.access.Packed_A404, align 8
; CHECK:  %p_tmp18406 = fmul double %tmp17_p_scalar_405, %arg3.s2a.reload399
; CHECK:  %polly.access.cast.Packed_B407 = bitcast [3072 x [8 x double]]* %Packed_B to double*
; CHECK  %245 = mul nsw i64 256, %polly.indvar101
; CHECK  %246 = add nsw i64 %245, %polly.indvar107
; CHECK  %polly.access.add.Packed_B408 = add nsw i64 0, %246
; CHECK  %polly.access.mul.Packed_B409 = mul nsw i64 %polly.access.add.Packed_B408, 8
; CHECK  %polly.access.add.Packed_B410 = add nsw i64 %polly.access.mul.Packed_B409, 0
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
  br label %bb14

bb14:                                             ; preds = %bb14, %bb9
  %tmp15 = phi i64 [ 0, %bb9 ], [ %tmp24, %bb14 ]
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
  br i1 %tmp25, label %bb14, label %bb26

bb26:                                             ; preds = %bb14
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
