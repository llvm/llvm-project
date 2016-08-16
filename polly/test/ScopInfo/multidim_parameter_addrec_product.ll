; RUN: opt %loadPolly -polly-scops -analyze \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
;    void foo(float *A, long *p) {
;      for (long i = 0; i < 100; i++)
;        for (long j = 0; j < 100; j++)
;          A[i * (*p) + j] += i + j;
;    }

; CHECK:  Invariant Accesses: {
; CHECK-NEXT:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [pval] -> { Stmt_bb5[i0, i1] -> MemRef_p[0] };
; CHECK-NEXT:          Execution Context: [pval] -> {  :  }
; CHECK-NEXT:  }

; CHECK: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:     [pval] -> { Stmt_bb5[i0, i1] -> MemRef_A[i0, i1] };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:     [pval] -> { Stmt_bb5[i0, i1] -> MemRef_A[i0, i1] };

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, i64* %p) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb16, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp17, %bb16 ]
  %exitcond1 = icmp ne i64 %i.0, 100
  br i1 %exitcond1, label %bb3, label %bb18

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb13, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp14, %bb13 ]
  %exitcond = icmp ne i64 %j.0, 100
  br i1 %exitcond, label %bb5, label %bb15

bb5:                                              ; preds = %bb4
  %tmp = add nuw nsw i64 %i.0, %j.0
  %tmp6 = sitofp i64 %tmp to float
  %pval = load i64, i64* %p, align 8
  %tmp8 = mul nsw i64 %i.0, %pval
  %tmp9 = add nsw i64 %tmp8, %j.0
  %tmp10 = getelementptr inbounds float, float* %A, i64 %tmp9
  %tmp11 = load float, float* %tmp10, align 4
  %tmp12 = fadd float %tmp11, %tmp6
  store float %tmp12, float* %tmp10, align 4
  br label %bb13

bb13:                                             ; preds = %bb5
  %tmp14 = add nuw nsw i64 %j.0, 1
  br label %bb4

bb15:                                             ; preds = %bb4
  br label %bb16

bb16:                                             ; preds = %bb15
  %tmp17 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb18:                                             ; preds = %bb2
  ret void
}
