; RUN: opt %loadPolly -S  -polly-process-unprofitable -polly-acc-mincompute=0 -polly-target=gpu  -polly-codegen-ppcg -polly-acc-codegen-managed-memory < %s | \
; RUN: FileCheck %s

; REQUIRES: pollyacc

;
;    #include <cuda_runtime.h>
;
;    static const int N = 45;
;
;    void copy(int *R, int *A) {
;      for (int i = 0; i < N; i++) {
;        R[i] = A[i] * 10;
;      }
;    }
;
;    int main() {
;      int *A, *R;
;
;      cudaMallocManaged((void **)(&A), sizeof(int) * N, cudaMemAttachGlobal);
;      cudaMallocManaged((void **)(&R), sizeof(int) * N, cudaMemAttachGlobal);
;
;      for (int i = 0; i < N; i++) {
;        A[i] = i;
;        R[i] = 0;
;      }
;      copy(R, A);
;
;      return 0;
;    }
;

; CHECK-NOT: polly_copyFromHostToDevice
; CHECK-NOT: polly_copyFromDeviceToHost
; CHECK-NOT: polly_freeDeviceMemory
; CHECK-NOT: polly_allocateMemoryForDevice

; CHECK:       %[[REGCTX:[0-9]+]] = call i8* @polly_initContextCUDA()
; CHECK-NEXT:  %[[REGCA:[0-9]+]] = bitcast i32* %A to i8*
; CHECK-NEXT:  %[[REGCR:[0-9]+]] = bitcast i32* %R to i8*
; CHECK-NEXT:  %[[REGGEP0:[0-9]+]] = getelementptr [2 x i8*], [2 x i8*]* %polly_launch_0_params, i64 0, i64 0
; CHECK-NEXT:  store i8* %[[REGCA]], i8** %polly_launch_0_param_0
; CHECK-NEXT:  %[[REGCP0:[0-9]+]] = bitcast i8** %polly_launch_0_param_0 to i8*
; CHECK-NEXT:  store i8* %[[REGCP0]], i8** %[[REGGEP0]]
; CHECK-NEXT:  %[[REGGEP1:[0-9]+]] = getelementptr [2 x i8*], [2 x i8*]* %polly_launch_0_params, i64 0, i64 1
; CHECK-NEXT:  store i8* %[[REGCR]], i8** %polly_launch_0_param_1
; CHECK-NEXT:  %[[REGCP1:[0-9]+]] = bitcast i8** %polly_launch_0_param_1 to i8*
; CHECK-NEXT:  store i8* %[[REGCP1]], i8** %[[REGGEP1]]
; CHECK-NEXT:  %[[REGKERNEL:[0-9]+]] = call i8* @polly_getKernel(i8* getelementptr inbounds ([852 x i8], [852 x i8]* @FUNC_copy_SCOP_0_KERNEL_0, i32 0, i32 0), i8* getelementptr inbounds ([26 x i8], [26 x i8]* @FUNC_copy_SCOP_0_KERNEL_0_name, i32 0, i32 0))
; CHECK-NEXT:  call void @polly_launchKernel(i8* %[[REGKERNEL]], i32 2, i32 1, i32 32, i32 1, i32 1, i8* %polly_launch_0_params_i8ptr)
; CHECK-NEXT:  call void @polly_freeKernel(i8* %[[REGKERNEL]])
; CHECK-NEXT:  call void @polly_synchronizeDevice()
; CHECK-NEXT:  call void @polly_freeContext(i8* %[[REGCTX]])

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @copy(i32* %R, i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 45
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %mul = mul nsw i32 %tmp, 10
  %arrayidx2 = getelementptr inbounds i32, i32* %R, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define i32 @main() {
entry:
  %A = alloca i32*, align 8
  %R = alloca i32*, align 8
  %tmp = bitcast i32** %A to i8**
  %call = call i32 @cudaMallocManaged(i8** nonnull %tmp, i64 180, i32 1) #2
  %tmp1 = bitcast i32** %R to i8**
  %call1 = call i32 @cudaMallocManaged(i8** nonnull %tmp1, i64 180, i32 1) #2
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 45
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp2 = load i32*, i32** %A, align 8
  %arrayidx = getelementptr inbounds i32, i32* %tmp2, i64 %indvars.iv
  %tmp3 = trunc i64 %indvars.iv to i32
  store i32 %tmp3, i32* %arrayidx, align 4
  %tmp4 = load i32*, i32** %R, align 8
  %arrayidx3 = getelementptr inbounds i32, i32* %tmp4, i64 %indvars.iv
  store i32 0, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %tmp5 = load i32*, i32** %R, align 8
  %tmp6 = load i32*, i32** %A, align 8
  call void @copy(i32* %tmp5, i32* %tmp6)
  ret i32 0
}

declare i32 @cudaMallocManaged(i8**, i64, i32) #1
