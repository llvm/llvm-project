; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-schedule \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=SCHED %s

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s -check-prefix=KERNEL-IR

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-asm \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s -check-prefix=KERNEL-ASM

; REQUIRES: pollyacc

; CHECK: Stmt_bb5
; CHECK-NEXT:       Domain :=
; CHECK-NEXT:           { Stmt_bb5[i0, i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 };
; CHECK-NEXT:       Schedule :=
; CHECK-NEXT:           { Stmt_bb5[i0, i1] -> [i0, i1] };
; CHECK-NEXT:       ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:           { Stmt_bb5[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:           { Stmt_bb5[i0, i1] -> MemRef_A[i0, i1] };

; SCHED: domain: "{ Stmt_bb5[i0, i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 }"
; SCHED-NEXT: child:
; SCHED-NEXT:   context: "{ [] }"
; SCHED-NEXT:   child:
; SCHED-NEXT:     extension: "{ [] -> from_device_MemRef_A[]; [] -> to_device_MemRef_A[] }"
; SCHED-NEXT:     child:
; SCHED-NEXT:       sequence:
; SCHED-NEXT:       - filter: "{ to_device_MemRef_A[] }"
; SCHED-NEXT:         child:
; SCHED-NEXT:           set:
; SCHED-NEXT:           - filter: "{ to_device_MemRef_A[] }"
; SCHED-NEXT:             child:
; SCHED-NEXT:               guard: "{ [] }"
; SCHED-NEXT:       - filter: "{ Stmt_bb5[i0, i1] }"
; SCHED-NEXT:         child:
; SCHED-NEXT:           guard: "{ [] }"
; SCHED-NEXT:           child:
; SCHED-NEXT:             mark: "kernel"
; SCHED-NEXT:             child:
; SCHED-NEXT:               context: "[b0, b1, t0, t1] -> { [] : 0 <= b0 <= 31 and 0 <= b1 <= 31 and 0 <= t0 <= 31 and 0 <= t1 <= 15 }"
; SCHED-NEXT:               child:
; SCHED-NEXT:                 filter: "[b0, b1] -> { Stmt_bb5[i0, i1] : -31 - 32b0 + i0 <= 8192*floor((i0)/8192) <= -32b0 + i0 and -31 - 32b1 + i1 <= 8192*floor((i1)/8192) <= -32b1 + i1 }"
; SCHED-NEXT:                 child:
; SCHED-NEXT:                   schedule: "[{ Stmt_bb5[i0, i1] -> [(floor((i0)/8192))] }, { Stmt_bb5[i0, i1] -> [(floor((i1)/8192))] }]"
; SCHED-NEXT:                   permutable: 1
; SCHED-NEXT:                   coincident: [ 1, 1 ]
; SCHED-NEXT:                   child:
; SCHED-NEXT:                     filter: "[t0, t1] -> { Stmt_bb5[i0, i1] : 32*floor((-t0 + i0)/32) = -t0 + i0 and 16*floor((-t1 + i1)/16) = -t1 + i1 and 0 <= t0 <= 31 and 0 <= t1 <= 15 }"
; SCHED-NEXT:                     child:
; SCHED-NEXT:                       schedule: "[{ Stmt_bb5[i0, i1] -> [(0)] }, { Stmt_bb5[i0, i1] -> [(floor((i1)/16) - 2*floor((i1)/32))] }]"
; SCHED-NEXT:                       permutable: 1
; SCHED-NEXT:                       coincident: [ 1, 1 ]
; SCHED-NEXT:       - filter: "{ from_device_MemRef_A[] }"
; SCHED-NEXT:         child:
; SCHED-NEXT:           set:
; SCHED-NEXT:           - filter: "{ from_device_MemRef_A[] }"
; SCHED-NEXT:             child:
; SCHED-NEXT:               guard: "{ [] }"

; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * (1024) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(16, 32);
; CODE-NEXT:     dim3 k0_dimGrid(32, 32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * (1024) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: for (int c3 = 0; c3 <= 1; c3 += 1)
; CODE-NEXT:   Stmt_bb5(32 * b0 + t0, 32 * b1 + t1 + 16 * c3);

; IR: polly.split_new_and_old:
; IR-NEXT:    br i1 true, label %polly.start, label %bb2

; IR: polly.start:
; IR-NEXT: br label %polly.acc.initialize

; IR: polly.acc.initialize:
; IR-NEXT:    [[GPUContext:%.*]] = call i8* @polly_initContext()
; IR-NEXT:    %p_dev_array_MemRef_A = call i8* @polly_allocateMemoryForDevice(i64 4194304)
; IR-NEXT:    [[HostPtr:%.*]] = bitcast [1024 x float]* %A to i8*
; IR-NEXT:    call void @polly_copyFromHostToDevice(i8* [[HostPtr]], i8* %p_dev_array_MemRef_A, i64 4194304)
; IR-NEXT:    [[DevPtr:%.*]]  = call i8* @polly_getDevicePtr(i8* %p_dev_array_MemRef_A)
; IR-NEXT:    store i8* [[DevPtr]], i8** %polly_launch_0_param_0
; IR-NEXT:    [[ParamSlot:%.*]] = getelementptr [1 x i8*], [1 x i8*]* %polly_launch_0_params, i64 0, i64 0
; IR-NEXT:    [[ParamTyped:%.*]] = bitcast i8** %polly_launch_0_param_0 to i8*
; IR-NEXT:    store i8* [[ParamTyped]], i8** [[ParamSlot]]
; IR-NEXT:    call i8* @polly_getKernel
; IR-NEXT:    call void @polly_launchKernel(i8* %5, i32 32, i32 32, i32 32, i32 16, i32 1, i8* %polly_launch_0_params_i8ptr)
; IR-NEXT:    call void @polly_freeKernel
; IR-NEXT:    [[HostPtr2:%.*]] = bitcast [1024 x float]* %A to i8*
; IR-NEXT:    call void @polly_copyFromDeviceToHost(i8* %p_dev_array_MemRef_A, i8* [[HostPtr2]], i64 4194304)
; IR-NEXT:    call void @polly_freeDeviceMemory(i8* %p_dev_array_MemRef_A)
; IR-NEXT:    call void @polly_freeContext(i8* [[GPUContext]])
; IR-NEXT:    br label %polly.exiting

; IR: polly.exiting:
; IR-NEXT:    br label %polly.merge_new_and_old

; KERNEL-IR-LABEL: define ptx_kernel void @kernel_0(i8* %MemRef_A) #0 {
; KERNEL-IR-NEXT: entry:
; KERNEL-IR-NEXT:   %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; KERNEL-IR-NEXT:   %b0 = zext i32 %0 to i64
; KERNEL-IR-NEXT:   %1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
; KERNEL-IR-NEXT:   %b1 = zext i32 %1 to i64
; KERNEL-IR-NEXT:   %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; KERNEL-IR-NEXT:   %t0 = zext i32 %2 to i64
; KERNEL-IR-NEXT:   %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; KERNEL-IR-NEXT:   %t1 = zext i32 %3 to i64
; KERNEL-IR-NEXT:   br label %polly.loop_preheader

; KERNEL-IR-LABEL: polly.loop_exit:                                  ; preds = %polly.stmt.bb5
; KERNEL-IR-NEXT:   ret void

; KERNEL-IR-LABEL: polly.loop_header:                                ; preds = %polly.stmt.bb5, %polly.loop_preheader
; KERNEL-IR-NEXT:   %polly.indvar = phi i64 [ 0, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.bb5 ]
; KERNEL-IR-NEXT:   %4 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %5 = add nsw i64 %4, %t0
; KERNEL-IR-NEXT:   %6 = mul nsw i64 32, %b1
; KERNEL-IR-NEXT:   %7 = add nsw i64 %6, %t1
; KERNEL-IR-NEXT:   %8 = mul nsw i64 16, %polly.indvar
; KERNEL-IR-NEXT:   %9 = add nsw i64 %7, %8
; KERNEL-IR-NEXT:   br label %polly.stmt.bb5

; KERNEL-IR-LABEL: polly.stmt.bb5:                                   ; preds = %polly.loop_header
; KERNEL-IR-NEXT:   %10 = mul i64 %9, %5
; KERNEL-IR-NEXT:   %p_tmp6 = sitofp i64 %10 to float
; KERNEL-IR-NEXT:   %polly.access.cast.MemRef_A = bitcast i8* %MemRef_A to float*
; KERNEL-IR-NEXT:   %11 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %12 = add nsw i64 %11, %t0
; KERNEL-IR-NEXT:   %polly.access.mul.MemRef_A = mul nsw i64 %12, 1024
; KERNEL-IR-NEXT:   %13 = mul nsw i64 32, %b1
; KERNEL-IR-NEXT:   %14 = add nsw i64 %13, %t1
; KERNEL-IR-NEXT:   %15 = mul nsw i64 16, %polly.indvar
; KERNEL-IR-NEXT:   %16 = add nsw i64 %14, %15
; KERNEL-IR-NEXT:   %polly.access.add.MemRef_A = add nsw i64 %polly.access.mul.MemRef_A, %16
; KERNEL-IR-NEXT:   %polly.access.MemRef_A = getelementptr float, float* %polly.access.cast.MemRef_A, i64 %polly.access.add.MemRef_A
; KERNEL-IR-NEXT:   %tmp8_p_scalar_ = load float, float* %polly.access.MemRef_A, align 4
; KERNEL-IR-NEXT:   %p_tmp9 = fadd float %tmp8_p_scalar_, %p_tmp6
; KERNEL-IR-NEXT:   %polly.access.cast.MemRef_A1 = bitcast i8* %MemRef_A to float*
; KERNEL-IR-NEXT:   %17 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %18 = add nsw i64 %17, %t0
; KERNEL-IR-NEXT:   %polly.access.mul.MemRef_A2 = mul nsw i64 %18, 1024
; KERNEL-IR-NEXT:   %19 = mul nsw i64 32, %b1
; KERNEL-IR-NEXT:   %20 = add nsw i64 %19, %t1
; KERNEL-IR-NEXT:   %21 = mul nsw i64 16, %polly.indvar
; KERNEL-IR-NEXT:   %22 = add nsw i64 %20, %21
; KERNEL-IR-NEXT:   %polly.access.add.MemRef_A3 = add nsw i64 %polly.access.mul.MemRef_A2, %22
; KERNEL-IR-NEXT:   %polly.access.MemRef_A4 = getelementptr float, float* %polly.access.cast.MemRef_A1, i64 %polly.access.add.MemRef_A3
; KERNEL-IR-NEXT:   store float %p_tmp9, float* %polly.access.MemRef_A4, align 4
; KERNEL-IR-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; KERNEL-IR-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar, 0
; KERNEL-IR-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; KERNEL-IR-LABEL: polly.loop_preheader:                             ; preds = %entry
; KERNEL-IR-NEXT:   br label %polly.loop_header

; KERNEL-IR: attributes #0 = { "polly.skip.fn" }

; KERNEL-ASM: .version 3.2
; KERNEL-ASM-NEXT: .target sm_30
; KERNEL-ASM-NEXT: .address_size 64

; KERNEL-ASM:   // .globl     kernel_0

; KERNEL-ASM: .visible .entry kernel_0(
; KERNEL-ASM-NEXT:   .param .u64 kernel_0_param_0
; KERNEL-ASM-NEXT: )

;    void double_parallel_loop(float A[][1024]) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          A[i][j] += i * j;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @double_parallel_loop([1024 x float]* %A) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb13, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp14, %bb13 ]
  %exitcond1 = icmp ne i64 %i.0, 1024
  br i1 %exitcond1, label %bb3, label %bb15

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb10, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp11, %bb10 ]
  %exitcond = icmp ne i64 %j.0, 1024
  br i1 %exitcond, label %bb5, label %bb12

bb5:                                              ; preds = %bb4
  %tmp = mul nuw nsw i64 %i.0, %j.0
  %tmp6 = sitofp i64 %tmp to float
  %tmp7 = getelementptr inbounds [1024 x float], [1024 x float]* %A, i64 %i.0, i64 %j.0
  %tmp8 = load float, float* %tmp7, align 4
  %tmp9 = fadd float %tmp8, %tmp6
  store float %tmp9, float* %tmp7, align 4
  br label %bb10

bb10:                                             ; preds = %bb5
  %tmp11 = add nuw nsw i64 %j.0, 1
  br label %bb4

bb12:                                             ; preds = %bb4
  br label %bb13

bb13:                                             ; preds = %bb12
  %tmp14 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb15:                                             ; preds = %bb2
  ret void
}
