; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s --check-prefix=SCOP
; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir -disable-output < %s | FileCheck %s --check-prefix=KERNEL-IR
; RUN: opt %loadPolly -S -polly-codegen-ppcg  < %s | FileCheck %s --check-prefix=HOST-IR

; Test that we do recognise and codegen a kernel that has intrinsics.

; REQUIRES: pollyacc

; Check that we model the kernel as a scop.
; SCOP:      Function: f
; SCOP-NEXT:       Region: %entry.split---%for.end

; Check that the intrinsic call is present in the kernel IR.
; KERNEL-IR:   %p_sqrt = tail call float @llvm.sqrt.f32(float %A.arr.i.val_p_scalar_)
; KERNEL-IR:   declare float @llvm.sqrt.f32(float)
; KERNEL-IR:   declare float @llvm.fabs.f32(float)


; Check that kernel launch is generated in host IR.
; the declare would not be generated unless a call to a kernel exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)


; void f(float *A, float *B, int N) {
;   for(int i = 0; i < N; i++) {
;       float tmp0 = A[i];
;       float tmp1 = sqrt(tmp1);
;       float tmp2 = fabs(tmp2);
;       float tmp3 = copysignf(tmp1, tmp2);
;       B[i] = tmp4;
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
  %sqrt = tail call float @llvm.sqrt.f32(float %A.arr.i.val)
  %fabs = tail call float @llvm.fabs.f32(float %sqrt);
  %copysign = tail call float @llvm.copysign.f32(float %sqrt, float %fabs);
  %B.arr.i = getelementptr inbounds float, float* %B, i64 %indvars.iv
  store float %copysign, float* %B.arr.i, align 4

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
declare float @llvm.sqrt.f32(float) #0
declare float @llvm.fabs.f32(float) #0
declare float @llvm.copysign.f32(float, float) #0

attributes #0 = { nounwind readnone }

