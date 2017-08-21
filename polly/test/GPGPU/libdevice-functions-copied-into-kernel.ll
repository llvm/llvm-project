; RUN: opt %loadPolly -analyze -polly-scops < %s \
; RUN: -polly-acc-libdevice=%S/Inputs/libdevice-functions-copied-into-kernel_libdevice.ll \
; RUN:     | FileCheck %s --check-prefix=SCOP
; RUN: opt %loadPolly -analyze -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -polly-acc-libdevice=%S/Inputs/libdevice-functions-copied-into-kernel_libdevice.ll \
; RUN:     < %s | FileCheck %s --check-prefix=KERNEL-IR
; RUN: opt %loadPolly -S -polly-codegen-ppcg  < %s \
; RUN: -polly-acc-libdevice=%S/Inputs/libdevice-functions-copied-into-kernel_libdevice.ll \
; RUN:     | FileCheck %s --check-prefix=HOST-IR

; Test that we do recognise and codegen a kernel that has functions that can
; be mapped to NVIDIA's libdevice

; REQUIRES: pollyacc

; Check that we model the kernel as a scop.
; SCOP:      Function: f
; SCOP-NEXT:       Region: %entry.split---%for.end

; Check that the intrinsic call is present in the kernel IR.
; KERNEL-IR:   %p_expf = tail call float @__nv_expf(float %A.arr.i.val_p_scalar_)
; KERNEL-IR:   %p_cosf = tail call float @__nv_cosf(float %p_expf)
; KERNEL-IR:   %p_logf = tail call float @__nv_logf(float %p_cosf)

; Check that kernel launch is generated in host IR.
; the declare would not be generated unless a call to a kernel exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)


; void f(float *A, float *B, int N) {
;   for(int i = 0; i < N; i++) {
;       float tmp0 = A[i];
;       float expf  = expf(tmp1);
;       cosf = cosf(expf);
;       logf = logf(cosf);
;       B[i] = logf;
;   }
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(float* %A, float* %B, i32 %N) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp1 = icmp sgt i32 %N, 0
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %A.arr.i = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %A.arr.i.val = load float, float* %A.arr.i, align 4
  ; Call to intrinsics that should be part of the kernel.
  %expf = tail call float @expf(float %A.arr.i.val)
  %cosf = tail call float @cosf(float %expf)
  %logf = tail call float @logf(float %cosf)
  %B.arr.i = getelementptr inbounds float, float* %B, i64 %indvars.iv
  store float %logf, float* %B.arr.i, align 4

  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %wide.trip.count = zext i32 %N to i64
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void
}

; Function Attrs: nounwind readnone
declare float @expf(float) #0
declare float @cosf(float) #0
declare float @logf(float) #0

attributes #0 = { nounwind readnone }

