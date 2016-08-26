; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -polly-acc-use-private \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg \
; RUN: -polly-acc-use-private \
; RUN: -disable-output -polly-acc-dump-kernel-ir < %s | \
; RUN: FileCheck -check-prefix=KERNEL %s

; REQUIRES: pollyacc

;    void add(float *A) {
;      for (long i = 0; i < 32; i++)
;        for (long j = 0; j < 10; j++)
;          A[i] += 1;
;    }

; CODE: # kernel0
; CODE: {
; CODE:     read(t0);
; CODE:     for (int c3 = 0; c3 <= 9; c3 += 1)
; CODE:       Stmt_bb5(t0, c3);
; CODE:     write(t0);
; CODE: }

; KERNEL: %private_array = alloca [1 x float]

; KERNEL:   %polly.access.cast.private_array = bitcast [1 x float]* %private_array to float*
; KERNEL-NEXT:   %polly.access.private_array = getelementptr float, float* %polly.access.cast.private_array, i64 0
; KERNEL-NEXT:   %polly.access.cast.MemRef_A = bitcast i8* %MemRef_A to float*
; KERNEL-NEXT:   %polly.access.MemRef_A = getelementptr float, float* %polly.access.cast.MemRef_A, i64 %t0
; KERNEL-NEXT:   %shared.read = load float, float* %polly.access.MemRef_A
; KERNEL-NEXT:   store float %shared.read, float* %polly.access.private_array

; KERNEL:   %polly.access.cast.private_array5 = bitcast [1 x float]* %private_array to float*
; KERNEL-NEXT:   %polly.access.private_array6 = getelementptr float, float* %polly.access.cast.private_array5, i64 0
; KERNEL-NEXT:   %polly.access.cast.MemRef_A7 = bitcast i8* %MemRef_A to float*
; KERNEL-NEXT:   %polly.access.MemRef_A8 = getelementptr float, float* %polly.access.cast.MemRef_A7, i64 %t0
; KERNEL-NEXT:   %shared.write = load float, float* %polly.access.private_array6
; KERNEL-NEXT:   store float %shared.write, float* %polly.access.MemRef_A8

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @add(float* %A) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb11, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  %exitcond1 = icmp ne i64 %i.0, 32
  br i1 %exitcond1, label %bb3, label %bb13

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb8, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp9, %bb8 ]
  %exitcond = icmp ne i64 %j.0, 10
  br i1 %exitcond, label %bb5, label %bb10

bb5:                                              ; preds = %bb4
  %tmp = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp6 = load float, float* %tmp, align 4
  %tmp7 = fadd float %tmp6, 1.000000e+00
  store float %tmp7, float* %tmp, align 4
  br label %bb8

bb8:                                              ; preds = %bb5
  %tmp9 = add nuw nsw i64 %j.0, 1
  br label %bb4

bb10:                                             ; preds = %bb4
  br label %bb11

bb11:                                             ; preds = %bb10
  %tmp12 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb13:                                             ; preds = %bb2
  ret void
}
