; RUN: opt %loadPolly -polly-codegen-ppcg \
; RUN: -polly-gpu-arch=spir64 \
; RUN: -polly-acc-dump-kernel-ir -polly-process-unprofitable -disable-output < %s | \
; RUN: FileCheck -check-prefix=I64 %s

; RUN: opt %loadPolly -polly-codegen-ppcg \
; RUN: -polly-gpu-arch=spir32 \
; RUN: -polly-acc-dump-kernel-ir -polly-process-unprofitable -disable-output < %s | \
; RUN: FileCheck -check-prefix=I32 %s

; REQUIRES: pollyacc

; This test case checks whether the openCl runtime functions (get_local_id/get_group_id) return the right types for 32 and 64bit devices.

; I32:      target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
; I32-NEXT: target triple = "spir-unknown-unknown"

; I32-LABEL: define spir_kernel void @FUNC_double_parallel_loop_SCOP_0_KERNEL_0(ptr addrspace(1) %MemRef_A) #0 !kernel_arg_addr_space !0 !kernel_arg_name !1 !kernel_arg_access_qual !1 !kernel_arg_type !1 !kernel_arg_type_qual !1 !kernel_arg_base_type !1 {
; I32-NEXT: entry:
; I32-NEXT:   %0 = call i32 @__gen_ocl_get_group_id0()
; I32-NEXT:   %__gen_ocl_get_group_id0 = zext i32 %0 to i64
; I32-NEXT:   %1 = call i32 @__gen_ocl_get_group_id1()
; I32-NEXT:   %__gen_ocl_get_group_id1 = zext i32 %1 to i64
; I32-NEXT:   %2 = call i32 @__gen_ocl_get_local_id0()
; I32-NEXT:   %__gen_ocl_get_local_id0 = zext i32 %2 to i64
; I32-NEXT:   %3 = call i32 @__gen_ocl_get_local_id1()
; I32-NEXT:   %__gen_ocl_get_local_id1 = zext i32 %3 to i64
; I32-NEXT:   br label %polly.loop_preheader

; I64:       target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
; I64-next:  target triple = "spir64-unknown-unknown"

; I64-LABEL: define spir_kernel void @FUNC_double_parallel_loop_SCOP_0_KERNEL_0(ptr addrspace(1) %MemRef_A) #0 !kernel_arg_addr_space !0 !kernel_arg_name !1 !kernel_arg_access_qual !1 !kernel_arg_type !1 !kernel_arg_type_qual !1 !kernel_arg_base_type !1 {
; I64-NEXT: entry:
; I64-NEXT:   %0 = call i64 @__gen_ocl_get_group_id0()
; I64-NEXT:   %1 = call i64 @__gen_ocl_get_group_id1()
; I64-NEXT:   %2 = call i64 @__gen_ocl_get_local_id0()
; I64-NEXT:   %3 = call i64 @__gen_ocl_get_local_id1()
; I64-NEXT:   br label %polly.loop_preheader


;    void double_parallel_loop(float A[][1024]) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          A[i][j] += i * j;
;    }
;

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @double_parallel_loop(ptr %A) {
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
  %tmp7 = getelementptr inbounds [1024 x float], ptr %A, i64 %i.0, i64 %j.0
  %tmp8 = load float, ptr %tmp7, align 4
  %tmp9 = fadd float %tmp8, %tmp6
  store float %tmp9, ptr %tmp7, align 4
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
