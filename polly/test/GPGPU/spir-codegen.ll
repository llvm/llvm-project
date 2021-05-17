; RUN: opt %loadPolly -polly-codegen-ppcg -polly-target=gpu \
; RUN: -polly-gpu-arch=spir32 \
; RUN: -polly-acc-dump-kernel-ir -polly-process-unprofitable -disable-output -enable-new-pm=0 < %s | \
; RUN: FileCheck %s

; REQUIRES: pollyacc

; CHECK:      target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
; CHECK-NEXT: target triple = "spir-unknown-unknown"

; CHECK-LABEL: define spir_kernel void @FUNC_double_parallel_loop_SCOP_0_KERNEL_0(i8 addrspace(1)* %MemRef_A) #0 !kernel_arg_addr_space !0 !kernel_arg_name !1 !kernel_arg_access_qual !1 !kernel_arg_type !1 !kernel_arg_type_qual !1 !kernel_arg_base_type !1 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call i32 @__gen_ocl_get_group_id0()
; CHECK-NEXT:   %__gen_ocl_get_group_id0 = zext i32 %0 to i64
; CHECK-NEXT:   %1 = call i32 @__gen_ocl_get_group_id1()
; CHECK-NEXT:   %__gen_ocl_get_group_id1 = zext i32 %1 to i64
; CHECK-NEXT:   %2 = call i32 @__gen_ocl_get_local_id0()
; CHECK-NEXT:   %__gen_ocl_get_local_id0 = zext i32 %2 to i64
; CHECK-NEXT:   %3 = call i32 @__gen_ocl_get_local_id1()
; CHECK-NEXT:   %__gen_ocl_get_local_id1 = zext i32 %3 to i64
; CHECK-NEXT:   br label %polly.loop_preheader

; CHECK-LABEL: polly.loop_exit:                                  ; preds = %polly.stmt.bb5
; CHECK-NEXT:   ret void

; CHECK-LABEL: polly.loop_header:                                ; preds = %polly.stmt.bb5, %polly.loop_preheader
; CHECK-NEXT:   %polly.indvar = phi i64 [ 0, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.bb5 ]
; CHECK-NEXT:   %4 = mul nsw i64 32, %__gen_ocl_get_group_id0
; CHECK-NEXT:   %5 = add nsw i64 %4, %__gen_ocl_get_local_id0
; CHECK-NEXT:   %6 = mul nsw i64 32, %__gen_ocl_get_group_id1
; CHECK-NEXT:   %7 = add nsw i64 %6, %__gen_ocl_get_local_id1
; CHECK-NEXT:   %8 = mul nsw i64 16, %polly.indvar
; CHECK-NEXT:   %9 = add nsw i64 %7, %8
; CHECK-NEXT:   br label %polly.stmt.bb5

; CHECK-LABEL: polly.stmt.bb5:                                   ; preds = %polly.loop_header
; CHECK-NEXT:   %10 = mul i64 %5, %9
; CHECK-NEXT:   %p_tmp6 = sitofp i64 %10 to float
; CHECK-NEXT:   %polly.access.cast.MemRef_A = bitcast i8 addrspace(1)* %MemRef_A to float addrspace(1)*
; CHECK-NEXT:   %11 = mul nsw i64 32, %__gen_ocl_get_group_id0
; CHECK-NEXT:   %12 = add nsw i64 %11, %__gen_ocl_get_local_id0
; CHECK-NEXT:   %polly.access.mul.MemRef_A = mul nsw i64 %12, 1024
; CHECK-NEXT:   %13 = mul nsw i64 32, %__gen_ocl_get_group_id1
; CHECK-NEXT:   %14 = add nsw i64 %13, %__gen_ocl_get_local_id1
; CHECK-NEXT:   %15 = mul nsw i64 16, %polly.indvar
; CHECK-NEXT:   %16 = add nsw i64 %14, %15
; CHECK-NEXT:   %polly.access.add.MemRef_A = add nsw i64 %polly.access.mul.MemRef_A, %16
; CHECK-NEXT:   %polly.access.MemRef_A = getelementptr float, float addrspace(1)* %polly.access.cast.MemRef_A, i64 %polly.access.add.MemRef_A
; CHECK-NEXT:   %tmp8_p_scalar_ = load float, float addrspace(1)* %polly.access.MemRef_A, align 4
; CHECK-NEXT:   %p_tmp9 = fadd float %tmp8_p_scalar_, %p_tmp6
; CHECK-NEXT:   %polly.access.cast.MemRef_A1 = bitcast i8 addrspace(1)* %MemRef_A to float addrspace(1)*
; CHECK-NEXT:   %17 = mul nsw i64 32, %__gen_ocl_get_group_id0
; CHECK-NEXT:   %18 = add nsw i64 %17, %__gen_ocl_get_local_id0
; CHECK-NEXT:   %polly.access.mul.MemRef_A2 = mul nsw i64 %18, 1024
; CHECK-NEXT:   %19 = mul nsw i64 32, %__gen_ocl_get_group_id1
; CHECK-NEXT:   %20 = add nsw i64 %19, %__gen_ocl_get_local_id1
; CHECK-NEXT:   %21 = mul nsw i64 16, %polly.indvar
; CHECK-NEXT:   %22 = add nsw i64 %20, %21
; CHECK-NEXT:   %polly.access.add.MemRef_A3 = add nsw i64 %polly.access.mul.MemRef_A2, %22
; CHECK-NEXT:   %polly.access.MemRef_A4 = getelementptr float, float addrspace(1)* %polly.access.cast.MemRef_A1, i64 %polly.access.add.MemRef_A3
; CHECK-NEXT:   store float %p_tmp9, float addrspace(1)* %polly.access.MemRef_A4, align 4
; CHECK-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; CHECK-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, 1
; CHECK-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; CHECK-LABEL: polly.loop_preheader:                             ; preds = %entry
; CHECK-NEXT:   br label %polly.loop_header

; CHECK: attributes #0 = { "polly.skip.fn" }

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
