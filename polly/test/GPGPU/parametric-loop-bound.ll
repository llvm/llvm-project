; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg \
; RUN: -S < %s | \
; RUN: FileCheck -check-prefix=IR %s

; REQUIRES: pollyacc

;    void foo(long A[], long n) {
;      for (long i = 0; i < n; i++)
;        A[i] += 100;
;    }

; CODE: if (n >= 1) {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (n) * sizeof(i64), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(n >= 1048546 ? 32768 : floord(n + 31, 32));
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A, n);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (n) * sizeof(i64), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: for (int c0 = 0; c0 <= (n - 32 * b0 - 1) / 1048576; c0 += 1)
; CODE-NEXT:   if (n >= 32 * b0 + t0 + 1048576 * c0 + 1)
; CODE-NEXT:     Stmt_bb2(32 * b0 + t0 + 1048576 * c0);

; IR: store i64 %n, i64* %polly_launch_0_param_1
; IR-NEXT: %8 = getelementptr [2 x i8*], [2 x i8*]* %polly_launch_0_params, i64 0, i64 1
; IR-NEXT: %9 = bitcast i64* %polly_launch_0_param_1 to i8*
; IR-NEXT: store i8* %9, i8** %8

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64* %A, i64 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %tmp = icmp slt i64 %i.0, %n
  br i1 %tmp, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp3 = getelementptr inbounds i64, i64* %A, i64 %i.0
  %tmp4 = load i64, i64* %tmp3, align 8
  %tmp5 = add nsw i64 %tmp4, 100
  store i64 %tmp5, i64* %tmp3, align 8
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
