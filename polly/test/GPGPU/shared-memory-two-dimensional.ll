; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -polly-acc-use-shared \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg \
; RUN: -polly-acc-use-shared \
; RUN: -disable-output -polly-acc-dump-kernel-ir < %s | \
; RUN: FileCheck -check-prefix=KERNEL %s

; REQUIRES: pollyacc

;    void foo(float A[], float b[][8]) {
;      for (long i = 0; i < 32; i++)
;        for (long j = 0; j < 16; j++)
;          for (long k = 0; k < 8; k++)
;            A[i] += j * k * b[j][k];
;    }


; CODE:      # kernel0
; CODE-NEXT: {
; CODE-NEXT:   if (t0 <= 7)
; CODE-NEXT:     for (int c0 = 0; c0 <= 15; c0 += 1)
; CODE-NEXT:       read(c0, t0);
; CODE-NEXT:   read(t0);
; CODE-NEXT:   sync0();
; CODE-NEXT:   for (int c3 = 0; c3 <= 15; c3 += 1)
; CODE-NEXT:     for (int c4 = 0; c4 <= 7; c4 += 1)
; CODE-NEXT:       Stmt_bb8(t0, c3, c4);
; CODE-NEXT:   sync1();
; CODE-NEXT:   write(t0);
; CODE-NEXT: }

; KERNEL: @shared_MemRef_b = internal addrspace(3) global [16 x [8 x float]] zeroinitializer, align 4

; KERNEL:        %polly.access.mul.MemRef_b = mul nsw i64 %polly.indvar, 8
; KERNEL-NEXT:   %polly.access.add.MemRef_b = add nsw i64 %polly.access.mul.MemRef_b, %t0
; KERNEL-NEXT:   %polly.access.MemRef_b = getelementptr float, float addrspace(1)* %polly.access.cast.MemRef_b, i64 %polly.access.add.MemRef_b
; KERNEL-NEXT:   %shared.read = load float, float addrspace(1)* %polly.access.MemRef_b
; KERNEL-NEXT:   store float %shared.read, float addrspace(3)* %polly.access.shared_MemRef_b


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, [8 x float]* %b) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb22, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp23, %bb22 ]
  %exitcond2 = icmp ne i64 %i.0, 32
  br i1 %exitcond2, label %bb4, label %bb24

bb4:                                              ; preds = %bb3
  br label %bb5

bb5:                                              ; preds = %bb19, %bb4
  %j.0 = phi i64 [ 0, %bb4 ], [ %tmp20, %bb19 ]
  %exitcond1 = icmp ne i64 %j.0, 16
  br i1 %exitcond1, label %bb6, label %bb21

bb6:                                              ; preds = %bb5
  br label %bb7

bb7:                                              ; preds = %bb16, %bb6
  %k.0 = phi i64 [ 0, %bb6 ], [ %tmp17, %bb16 ]
  %exitcond = icmp ne i64 %k.0, 8
  br i1 %exitcond, label %bb8, label %bb18

bb8:                                              ; preds = %bb7
  %tmp = mul nuw nsw i64 %j.0, %k.0
  %tmp9 = sitofp i64 %tmp to float
  %tmp10 = getelementptr inbounds [8 x float], [8 x float]* %b, i64 %j.0, i64 %k.0
  %tmp11 = load float, float* %tmp10, align 4
  %tmp12 = fmul float %tmp9, %tmp11
  %tmp13 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp14 = load float, float* %tmp13, align 4
  %tmp15 = fadd float %tmp14, %tmp12
  store float %tmp15, float* %tmp13, align 4
  br label %bb16

bb16:                                             ; preds = %bb8
  %tmp17 = add nuw nsw i64 %k.0, 1
  br label %bb7

bb18:                                             ; preds = %bb7
  br label %bb19

bb19:                                             ; preds = %bb18
  %tmp20 = add nuw nsw i64 %j.0, 1
  br label %bb5

bb21:                                             ; preds = %bb5
  br label %bb22

bb22:                                             ; preds = %bb21
  %tmp23 = add nuw nsw i64 %i.0, 1
  br label %bb3

bb24:                                             ; preds = %bb3
  ret void
}
