; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg \
; RUN: -S < %s | \
; RUN: FileCheck -check-prefix=IR %s

; RUN: opt %loadPolly -polly-codegen-ppcg \
; RUN: -disable-output -polly-acc-dump-kernel-ir < %s | \
; RUN: FileCheck -check-prefix=KERNEL %s

; REQUIRES: pollyacc,nvptx

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; KERNEL: define ptx_kernel void @kernel_0(i8* %MemRef_A, float %MemRef_b)

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A, MemRef_b);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(float A[], float b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @float(float* %A, float %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp3 = load float, float* %tmp, align 4
  %tmp4 = fadd float %tmp3, %b
  store float %tmp4, float* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

; KERNEL: define ptx_kernel void @kernel_0(i8* %MemRef_A, double %MemRef_b)
; KERNEL-NEXT: entry:
; KERNEL-NEXT:   %b.s2a = alloca double
; KERNEL-NEXT:   store double %MemRef_b, double* %b.s2a

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(double), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A, MemRef_b);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(double), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(double A[], double b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @double(double* %A, double %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds double, double* %A, i64 %i.0
  %tmp3 = load double, double* %tmp, align 4
  %tmp4 = fadd double %tmp3, %b
  store double %tmp4, double* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(i1), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(i1), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(i1 A[], i1 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i1(i1* %A, i1 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i1, i1* %A, i64 %i.0
  %tmp3 = load i1, i1* %tmp, align 4
  %tmp4 = add i1 %tmp3, %b
  store i1 %tmp4, i1* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(i3), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(i3), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(i3 A[], i3 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i3(i3* %A, i3 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i3, i3* %A, i64 %i.0
  %tmp3 = load i3, i3* %tmp, align 4
  %tmp4 = add i3 %tmp3, %b
  store i3 %tmp4, i3* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(i8), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(i8), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(i8 A[], i32 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i8(i8* %A, i8 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i8, i8* %A, i64 %i.0
  %tmp3 = load i8, i8* %tmp, align 4
  %tmp4 = add i8 %tmp3, %b
  store i8 %tmp4, i8* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

; IR-LABEL: @i8

; IR: [[REGA:%.+]] = call i8* @polly_getDevicePtr(i8* %p_dev_array_MemRef_A)
; IR-NEXT: [[REGB:%.+]] = getelementptr [2 x i8*], [2 x i8*]* %polly_launch_0_params, i64 0, i64 0
; IR-NEXT: store i8* [[REGA:%.+]], i8** %polly_launch_0_param_0
; IR-NEXT: [[REGC:%.+]] = bitcast i8** %polly_launch_0_param_0 to i8*
; IR-NEXT: store i8* [[REGC]], i8** [[REGB]]
; IR-NEXT: store i8 %b, i8* %polly_launch_0_param_1
; IR-NEXT: [[REGD:%.+]] = getelementptr [2 x i8*], [2 x i8*]* %polly_launch_0_params, i64 0, i64 1
; IR-NEXT: store i8* %polly_launch_0_param_1, i8** [[REGD]]

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(i32), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(i32), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(i32 A[], i32 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i32(i32* %A, i32 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32, i32* %A, i64 %i.0
  %tmp3 = load i32, i32* %tmp, align 4
  %tmp4 = add i32 %tmp3, %b
  store i32 %tmp4, i32* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(i60), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(i60), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(i60 A[], i60 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i60(i60* %A, i60 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i60, i60* %A, i64 %i.0
  %tmp3 = load i60, i60* %tmp, align 4
  %tmp4 = add i60 %tmp3, %b
  store i60 %tmp4, i60* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(i64), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(i64), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(i64 A[], i64 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i64(i64* %A, i64 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i64, i64* %A, i64 %i.0
  %tmp3 = load i64, i64* %tmp, align 4
  %tmp4 = add i64 %tmp3, %b
  store i64 %tmp4, i64* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}
