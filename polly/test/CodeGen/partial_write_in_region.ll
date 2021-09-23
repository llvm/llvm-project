; RUN: opt %loadPolly -polly-import-jscop \
; RUN: -polly-import-jscop-postfix=transformed -polly-codegen \
; RUN: -verify-dom-info \
; RUN: -S < %s | FileCheck %s
;
;    void foo(long A[], float B[], float C[]) {
;      for (long i = 0; i < 1024; i++) {
;        if (A[i]) {
; S:         B[i]++;
; T:         C[42] = 1;
;        }
;      }
;    }

; CHECK: polly.stmt.bb5:                                   ; preds = %polly.stmt.bb2
; CHECK-NEXT:   %scevgep10 = getelementptr float, float* %B, i64 %polly.indvar
; CHECK-NEXT:   %tmp7_p_scalar_ = load float, float* %scevgep10
; CHECK-NEXT:   %p_tmp8 = fadd float %tmp7_p_scalar_, 1.000000e+00
; CHECK-NEXT:   %24 = icmp sle i64 %polly.indvar, 9
; CHECK-NEXT:   %polly.Stmt_bb2__TO__bb9_MayWrite2.cond = icmp ne i1 %24, false
; CHECK-NEXT:   br i1 %polly.Stmt_bb2__TO__bb9_MayWrite2.cond, label %polly.stmt.bb5.Stmt_bb2__TO__bb9_MayWrite2.partial, label %polly.stmt.bb5.cont

; CHECK: polly.stmt.bb5.Stmt_bb2__TO__bb9_MayWrite2.partial: ; preds = %polly.stmt.bb5
; CHECK-NEXT:   %polly.access.B11 = getelementptr float, float* %B, i64 %polly.indvar
; CHECK-NEXT:   store float %p_tmp8, float* %polly.access.B11
; CHECK-NEXT:   br label %polly.stmt.bb5.cont

; CHECK: polly.stmt.bb5.cont:                              ; preds = %polly.stmt.bb5, %polly.stmt.bb5.Stmt_bb2__TO__bb9_MayWrite2.partial
; CHECK-NEXT:   br label %polly.stmt.bb9b

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @partial_write_in_region(i64* %A, float* %B, float* %C) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb10, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp11, %bb10 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb12

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i64, i64* %A, i64 %i.0
  %tmp3 = load i64, i64* %tmp, align 8
  %tmp4 = icmp eq i64 %tmp3, 0
  br i1 %tmp4, label %bb9, label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp7 = load float, float* %tmp6, align 4
  %tmp8 = fadd float %tmp7, 1.000000e+00
  store float %tmp8, float* %tmp6, align 4
  br label %bb9b

bb9b:
  store float 42.0, float* %C
  br label %bb9

bb9:                                              ; preds = %bb2, %bb5
  br label %bb10

bb10:                                             ; preds = %bb9
  %tmp11 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb12:                                             ; preds = %bb1
  ret void
}
