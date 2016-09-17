; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s -check-prefix=KERNEL-IR
;
; REQUIRES: pollyacc
;
; #include <stdio.h>
;
; float foo(float A[]) {
;   float sum = 0;
;
;   for (long i = 0; i < 32; i++)
;     A[i] = i;
;
;   for (long i = 0; i < 32; i++)
;     A[i] += i;
;
;   for (long i = 0; i < 32; i++)
;     sum += A[i];
;
;   return sum;
; }
;
; int main() {
;   float A[32];
;   float sum = foo(A);
;   printf("%f\n", sum);
; }

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(1);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   {
; CODE-NEXT:     dim3 k1_dimBlock;
; CODE-NEXT:     dim3 k1_dimGrid;
; CODE-NEXT:     kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_MemRef_sum_0__phi);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   for (int c0 = 0; c0 <= 32; c0 += 1) {
; CODE-NEXT:     {
; CODE-NEXT:       dim3 k2_dimBlock;
; CODE-NEXT:       dim3 k2_dimGrid;
; CODE-NEXT:       kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_MemRef_sum_0__phi, dev_MemRef_sum_0, c0);
; CODE-NEXT:       cudaCheckKernel();
; CODE-NEXT:     }

; CODE:     if (c0 <= 31)
; CODE-NEXT:       {
; CODE-NEXT:         dim3 k3_dimBlock;
; CODE-NEXT:         dim3 k3_dimGrid;
; CODE-NEXT:         kernel3 <<<k3_dimGrid, k3_dimBlock>>> (dev_MemRef_A, dev_MemRef_sum_0__phi, dev_MemRef_sum_0, c0);
; CODE-NEXT:         cudaCheckKernel();
; CODE-NEXT:       }

; CODE:   }
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (32) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(&MemRef_sum_0__phi, dev_MemRef_sum_0__phi, sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(&MemRef_sum_0, dev_MemRef_sum_0, sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: {
; CODE-NEXT:   Stmt_bb4(t0);
; CODE-NEXT:   Stmt_bb10(t0);
; CODE-NEXT: }

; CODE: # kernel1
; CODE-NEXT: Stmt_bb17();

; CODE: # kernel2
; CODE-NEXT: Stmt_bb18(c0);

; CODE: # kernel3
; CODE-NEXT: Stmt_bb20(c0);

; KERNEL-IR:       store float %p_tmp23, float* %sum.0.phiops
; KERNEL-IR-NEXT:  [[REGA:%.+]] = bitcast i8* %MemRef_sum_0__phi to float*
; KERNEL-IR-NEXT:  [[REGB:%.+]] = load float, float* %sum.0.phiops
; KERNEL-IR-NEXT:  store float [[REGB]], float* [[REGA]]
; KERNEL-IR-NEXT:  [[REGC:%.+]] = bitcast i8* %MemRef_sum_0 to float*
; KERNEL-IR-NEXT:  [[REGD:%.+]] = load float, float* %sum.0.s2a
; KERNEL-IR-NEXT:  store float [[REGD]], float* [[REGC]]
; KERNEL-IR-NEXT:  ret void

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

define float @foo(float* %A) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond2 = icmp ne i64 %i.0, 32
  br i1 %exitcond2, label %bb4, label %bb8

bb4:                                              ; preds = %bb3
  %tmp = sitofp i64 %i.0 to float
  %tmp5 = getelementptr inbounds float, float* %A, i64 %i.0
  store float %tmp, float* %tmp5, align 4
  br label %bb6

bb6:                                              ; preds = %bb4
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb3

bb8:                                              ; preds = %bb3
  br label %bb9

bb9:                                              ; preds = %bb15, %bb8
  %i1.0 = phi i64 [ 0, %bb8 ], [ %tmp16, %bb15 ]
  %exitcond1 = icmp ne i64 %i1.0, 32
  br i1 %exitcond1, label %bb10, label %bb17

bb10:                                             ; preds = %bb9
  %tmp11 = sitofp i64 %i1.0 to float
  %tmp12 = getelementptr inbounds float, float* %A, i64 %i1.0
  %tmp13 = load float, float* %tmp12, align 4
  %tmp14 = fadd float %tmp13, %tmp11
  store float %tmp14, float* %tmp12, align 4
  br label %bb15

bb15:                                             ; preds = %bb10
  %tmp16 = add nuw nsw i64 %i1.0, 1
  br label %bb9

bb17:                                             ; preds = %bb9
  br label %bb18

bb18:                                             ; preds = %bb20, %bb17
  %sum.0 = phi float [ 0.000000e+00, %bb17 ], [ %tmp23, %bb20 ]
  %i2.0 = phi i64 [ 0, %bb17 ], [ %tmp24, %bb20 ]
  %exitcond = icmp ne i64 %i2.0, 32
  br i1 %exitcond, label %bb19, label %bb25

bb19:                                             ; preds = %bb18
  br label %bb20

bb20:                                             ; preds = %bb19
  %tmp21 = getelementptr inbounds float, float* %A, i64 %i2.0
  %tmp22 = load float, float* %tmp21, align 4
  %tmp23 = fadd float %sum.0, %tmp22
  %tmp24 = add nuw nsw i64 %i2.0, 1
  br label %bb18

bb25:                                             ; preds = %bb18
  %sum.0.lcssa = phi float [ %sum.0, %bb18 ]
  ret float %sum.0.lcssa
}

define i32 @main() {
bb:
  %A = alloca [32 x float], align 16
  %tmp = getelementptr inbounds [32 x float], [32 x float]* %A, i64 0, i64 0
  %tmp1 = call float @foo(float* %tmp)
  %tmp2 = fpext float %tmp1 to double
  %tmp3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), double %tmp2) #2
  ret i32 0
}

declare i32 @printf(i8*, ...) #1

