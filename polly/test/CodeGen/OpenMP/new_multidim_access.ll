; RUN: opt %loadPolly -polly-print-import-jscop \
; RUN:                -disable-output < %s | FileCheck %s

; RUN: opt %loadPolly -polly-import-jscop \
; RUN:                -polly-codegen -S < %s \
; RUN:                -polly-parallel \
; RUN:                | FileCheck %s -check-prefix=IR

;    void new_multidim_access(long n, long m, float A[][m]) {
;      for (long i = 0; i < n; i++)
;        for (long j = 0; j < 100; j++)
;          A[i][2 * j] += i + j;
;    }

; CHECK:  ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:         [n, m] -> { Stmt_bb4[i0, i1] -> MemRef_A[i0, 2i1] };
; CHECK:    new: [n, m] -> { Stmt_bb4[i0, i1] -> MemRef_A[i0, 13 + i1] };
; CHECK:  MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:         [n, m] -> { Stmt_bb4[i0, i1] -> MemRef_A[i0, 2i1] };
; CHECK:    new: [n, m] -> { Stmt_bb4[i0, i1] -> MemRef_A[i0, 43 + i1] };

; IR: %polly.access.mul.polly.subfunc.arg.A = mul nsw i64 %polly.indvar, %polly.subfunc.arg.m
; IR: %6 = add nsw i64 %polly.indvar4, 13
; IR: %polly.access.add.polly.subfunc.arg.A = add nsw i64 %polly.access.mul.polly.subfunc.arg.A, %6
; IR: %polly.access.polly.subfunc.arg.A = getelementptr float, ptr %polly.subfunc.arg.A, i64 %polly.access.add.polly.subfunc.arg.A
; IR: %tmp10_p_scalar_ = load float, ptr %polly.access.polly.subfunc.arg.A, align 4, !alias.scope !0, !noalias !3, !llvm.access.group !4

; IR: %polly.access.mul.polly.subfunc.arg.A7 = mul nsw i64 %polly.indvar, %polly.subfunc.arg.m
; IR: %7 = add nsw i64 %polly.indvar4, 43
; IR: %polly.access.add.polly.subfunc.arg.A8 = add nsw i64 %polly.access.mul.polly.subfunc.arg.A7, %7
; IR: %polly.access.polly.subfunc.arg.A9 = getelementptr float, ptr %polly.subfunc.arg.A, i64 %polly.access.add.polly.subfunc.arg.A8
; IR: store float %p_tmp11, ptr %polly.access.polly.subfunc.arg.A9, align 4, !alias.scope !0, !noalias !3, !llvm.access.group !4
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @new_multidim_access(i64 %n, i64 %m, ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb15, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp16, %bb15 ]
  %tmp = icmp slt i64 %i.0, %n
  br i1 %tmp, label %bb2, label %bb17

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb12, %bb2
  %j.0 = phi i64 [ 0, %bb2 ], [ %tmp13, %bb12 ]
  %exitcond = icmp ne i64 %j.0, 100
  br i1 %exitcond, label %bb4, label %bb14

bb4:                                              ; preds = %bb3
  %tmp5 = add nsw i64 %i.0, %j.0
  %tmp6 = sitofp i64 %tmp5 to float
  %tmp7 = shl nsw i64 %j.0, 1
  %tmp8 = mul nsw i64 %i.0, %m
  %.sum = add i64 %tmp8, %tmp7
  %tmp9 = getelementptr inbounds float, ptr %A, i64 %.sum
  %tmp10 = load float, ptr %tmp9, align 4
  %tmp11 = fadd float %tmp10, %tmp6
  store float %tmp11, ptr %tmp9, align 4
  br label %bb12

bb12:                                             ; preds = %bb4
  %tmp13 = add nsw i64 %j.0, 1
  br label %bb3

bb14:                                             ; preds = %bb3
  br label %bb15

bb15:                                             ; preds = %bb14
  %tmp16 = add nsw i64 %i.0, 1
  br label %bb1

bb17:                                             ; preds = %bb1
  ret void
}
