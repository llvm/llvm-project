; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=KERNEL-IR %s

; REQUIRES: pollyacc

;    void kernel_params_scop_parameter(float A[], long n) {
;      for (long i = 0; i < n; i++)
;        A[i] += 42;
;    }

; KERNEL-IR: define ptx_kernel void @FUNC_kernel_params_scop_parameter_SCOP_0_KERNEL_0(i8 addrspace(1)* %MemRef_A, i64 %n)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @kernel_params_scop_parameter(float* %A, i64 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %tmp = icmp slt i64 %i.0, %n
  br i1 %tmp, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp3 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, 4.200000e+01
  store float %tmp5, float* %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
