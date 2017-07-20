; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; REQUIRES: pollyacc

;    void foo(float A[]) {
;      for (long i = 0; i < 128; i++)
;        A[i] += i;
;
;      for (long i = 0; i < 128; i++)
;        for (long j = 0; j < 128; j++)
;          A[42] += i + j;
;    }

; CODE:        cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (128) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(4);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:            {
; CODE-NEXT:         dim3 k1_dimBlock;
; CODE-NEXT:         dim3 k1_dimGrid;
; CODE-NEXT:         kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:         cudaCheckKernel();
; CODE-NEXT:       }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (128) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT: cudaCheckReturn(cudaFree(dev_MemRef_A));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb4(32 * b0 + t0);

; CODE: # kernel1
; CODE-NEXT: for (int c0 = 0; c0 <= 127; c0 += 1)
; CODE-NEXT:   for (int c1 = 0; c1 <= 127; c1 += 1)
; CODE-NEXT:     Stmt_bb14(c0, c1);


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb8, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp9, %bb8 ]
  %exitcond2 = icmp ne i64 %i.0, 128
  br i1 %exitcond2, label %bb4, label %bb10

bb4:                                              ; preds = %bb3
  %tmp = sitofp i64 %i.0 to float
  %tmp5 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp6 = load float, float* %tmp5, align 4
  %tmp7 = fadd float %tmp6, %tmp
  store float %tmp7, float* %tmp5, align 4
  br label %bb8

bb8:                                              ; preds = %bb4
  %tmp9 = add nuw nsw i64 %i.0, 1
  br label %bb3

bb10:                                             ; preds = %bb3
  br label %bb11

bb11:                                             ; preds = %bb23, %bb10
  %i1.0 = phi i64 [ 0, %bb10 ], [ %tmp24, %bb23 ]
  %exitcond1 = icmp ne i64 %i1.0, 128
  br i1 %exitcond1, label %bb12, label %bb25

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb20, %bb12
  %j.0 = phi i64 [ 0, %bb12 ], [ %tmp21, %bb20 ]
  %exitcond = icmp ne i64 %j.0, 128
  br i1 %exitcond, label %bb14, label %bb22

bb14:                                             ; preds = %bb13
  %tmp15 = add nuw nsw i64 %i1.0, %j.0
  %tmp16 = sitofp i64 %tmp15 to float
  %tmp17 = getelementptr inbounds float, float* %A, i64 42
  %tmp18 = load float, float* %tmp17, align 4
  %tmp19 = fadd float %tmp18, %tmp16
  store float %tmp19, float* %tmp17, align 4
  br label %bb20

bb20:                                             ; preds = %bb14
  %tmp21 = add nuw nsw i64 %j.0, 1
  br label %bb13

bb22:                                             ; preds = %bb13
  br label %bb23

bb23:                                             ; preds = %bb22
  %tmp24 = add nuw nsw i64 %i1.0, 1
  br label %bb11

bb25:                                             ; preds = %bb11
  ret void
}
