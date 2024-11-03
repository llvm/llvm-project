; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR
;
; REQUIRES: pollyacc

; CODE:      cudaCheckReturn(cudaMemcpy(dev_MemRef_B, MemRef_B, (16) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT: cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (8) * sizeof(float), cudaMemcpyHostToDevice));

; CODE:          dim3 k0_dimBlock(8);
; CODE-NEXT:     dim3 k0_dimGrid(1);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_B);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:        {
; CODE-NEXT:     dim3 k1_dimBlock(8);
; CODE-NEXT:     dim3 k1_dimGrid(1);
; CODE-NEXT:     kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_B, dev_MemRef_B, (16) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (8) * sizeof(float), cudaMemcpyDeviceToHost));

; CODE: # kernel0
; CODE-NEXT: Stmt_bb3(t0);

; CODE: # kernel1
; CODE-NEXT: Stmt_bb11(t0);

; IR:       %p_dev_array_MemRef_B = call ptr @polly_allocateMemoryForDevice(i64 32)
; IR-NEXT:  %p_dev_array_MemRef_A = call ptr @polly_allocateMemoryForDevice(i64 32)
; IR-NEXT:  [[REG0:%.+]] = getelementptr float, ptr %B, i64 8
; IR-NEXT:  call void @polly_copyFromHostToDevice(ptr [[REG0]], ptr %p_dev_array_MemRef_B, i64 32)

; IR:      [[REGA:%.+]] = call ptr @polly_getDevicePtr(ptr %p_dev_array_MemRef_B)
; IR-NEXT: [[REGC:%.+]]  = getelementptr float, ptr [[REGA]], i64 -8

;    void foo(float A[], float B[]) {
;      for (long i = 0; i < 8; i++)
;        B[i + 8] *= 4;
;
;      for (long i = 0; i < 8; i++)
;        A[i] *= 12;
;    }
;
;    #ifdef OUTPUT
;    int main() {
;      float A[16];
;
;      for (long i = 0; i < 16; i++) {
;        __sync_synchronize();
;        A[i] = i;
;      }
;
;      foo(A, A);
;
;      float sum = 0;
;      for (long i = 0; i < 16; i++) {
;        __sync_synchronize();
;        sum += A[i];
;      }
;
;      printf("%f\n", sum);
;    }
;    #endif
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %A, ptr %B) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb7, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp8, %bb7 ]
  %exitcond1 = icmp ne i64 %i.0, 8
  br i1 %exitcond1, label %bb3, label %bb9

bb3:                                              ; preds = %bb2
  %tmp = add nuw nsw i64 %i.0, 8
  %tmp4 = getelementptr inbounds float, ptr %B, i64 %tmp
  %tmp5 = load float, ptr %tmp4, align 4
  %tmp6 = fmul float %tmp5, 4.000000e+00
  store float %tmp6, ptr %tmp4, align 4
  br label %bb7

bb7:                                              ; preds = %bb3
  %tmp8 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb9:                                              ; preds = %bb2
  br label %bb10

bb10:                                             ; preds = %bb15, %bb9
  %i1.0 = phi i64 [ 0, %bb9 ], [ %tmp16, %bb15 ]
  %exitcond = icmp ne i64 %i1.0, 8
  br i1 %exitcond, label %bb11, label %bb17

bb11:                                             ; preds = %bb10
  %tmp12 = getelementptr inbounds float, ptr %A, i64 %i1.0
  %tmp13 = load float, ptr %tmp12, align 4
  %tmp14 = fmul float %tmp13, 1.200000e+01
  store float %tmp14, ptr %tmp12, align 4
  br label %bb15

bb15:                                             ; preds = %bb11
  %tmp16 = add nuw nsw i64 %i1.0, 1
  br label %bb10

bb17:                                             ; preds = %bb10
  ret void
}
