; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: not FileCheck %s -check-prefix=KERNEL-IR

; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR

; REQUIRES: pollyacc
;
;    void foo(long A[1024], long B[1024]) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += (B[i] + (long)&B[i]);
;    }

; This kernel loads/stores a pointer address we model. This is a rare case,
; were we still lack proper code-generation support. We check here that we
; detect the invalid IR and bail out gracefully.

; CODE:        cudaCheckReturn(cudaMemcpy(dev_MemRef_B, MemRef_B, (1024) * sizeof(i64), cudaMemcpyHostToDevice));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(i64), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_B, dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(i64), cudaMemcpyDeviceToHost));

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR

; KERNEL-IR: kernel

; IR: br i1 false, label %polly.start, label %bb1

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %A, ptr %B) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb10, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp11, %bb10 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb12

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i64, ptr %B, i64 %i.0
  %tmp3 = load i64, ptr %tmp, align 8
  %tmp4 = getelementptr inbounds i64, ptr %B, i64 %i.0
  %tmp5 = ptrtoint ptr %tmp4 to i64
  %tmp6 = add nsw i64 %tmp3, %tmp5
  %tmp7 = getelementptr inbounds i64, ptr %A, i64 %i.0
  %tmp8 = load i64, ptr %tmp7, align 8
  %tmp9 = add nsw i64 %tmp8, %tmp6
  store i64 %tmp9, ptr %tmp7, align 8
  br label %bb10

bb10:                                             ; preds = %bb2
  %tmp11 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb12:                                             ; preds = %bb1
  ret void
}
