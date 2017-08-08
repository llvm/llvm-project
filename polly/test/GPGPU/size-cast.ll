; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR

; REQUIRES: pollyacc

; This test case ensures that we properly sign-extend the types we are using.

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: if (arg >= 1 && arg1 == 0) {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_arg3, MemRef_arg3, (arg) * sizeof(double), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(arg >= 1048546 ? 32768 : floord(arg + 31, 32));
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_arg3, dev_MemRef_arg2, arg, arg1);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_arg2, dev_MemRef_arg2, (arg) * sizeof(double), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: for (int c0 = 0; c0 <= (arg - 32 * b0 - 1) / 1048576; c0 += 1)
; CODE-NEXT:   if (arg >= 32 * b0 + t0 + 1048576 * c0 + 1)
; CODE-NEXT:     Stmt_bb6(0, 32 * b0 + t0 + 1048576 * c0);

; IR:        call i8* @polly_initContextCUDA()
; IR-NEXT:   sext i32 %arg to i64
; IR-NEXT:   mul i64
; IR-NEXT:   @polly_allocateMemoryForDevice

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hoge(i32 %arg, i32 %arg1, [1000 x double]* %arg2, double* %arg3) {
bb:
  br label %bb4

bb4:                                              ; preds = %bb13, %bb
  br label %bb6

bb5:                                              ; preds = %bb13
  ret void

bb6:                                              ; preds = %bb6, %bb4
  %tmp = phi i64 [ 0, %bb4 ], [ %tmp10, %bb6 ]
  %tmp7 = getelementptr inbounds double, double* %arg3, i64 %tmp
  %tmp8 = load double, double* %tmp7, align 8
  %tmp9 = getelementptr inbounds [1000 x double], [1000 x double]* %arg2, i64 0, i64 %tmp
  store double undef, double* %tmp9, align 8
  %tmp10 = add nuw nsw i64 %tmp, 1
  %tmp11 = zext i32 %arg to i64
  %tmp12 = icmp ne i64 %tmp10, %tmp11
  br i1 %tmp12, label %bb6, label %bb13

bb13:                                             ; preds = %bb6
  %tmp14 = zext i32 %arg1 to i64
  %tmp15 = icmp ne i64 0, %tmp14
  br i1 %tmp15, label %bb4, label %bb5
}
