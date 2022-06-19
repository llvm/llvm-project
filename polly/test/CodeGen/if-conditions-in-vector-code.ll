; RUN: opt %loadPolly -polly-vectorizer=polly -polly-print-ast -disable-output < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadPolly -polly-vectorizer=polly -polly-codegen -S < %s | FileCheck %s
;
;    void foo(float *A) {
;      for (long i = 0; i < 16; i++) {
;        if (i % 2)
;          A[i] += 2;
;        if (i % 3)
;          A[i] += 3;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; AST: #pragma simd
; AST: #pragma known-parallel
; AST: for (int c0 = 0; c0 <= 15; c0 += 1) {
; AST:   if ((c0 + 1) % 2 == 0)
; AST:     Stmt_bb4(c0);
; AST:   if (c0 % 3 >= 1)
; AST:     Stmt_bb11(c0);
; AST: }

; CHECK: polly.split_new_and_old

define void @foo(float* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb16, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp17, %bb16 ]
  %exitcond = icmp ne i64 %i.0, 16
  br i1 %exitcond, label %bb2, label %bb18

bb2:                                              ; preds = %bb1
  %tmp = srem i64 %i.0, 2
  %tmp3 = icmp eq i64 %tmp, 0
  br i1 %tmp3, label %bb8, label %bb4

bb4:                                              ; preds = %bb2
  %tmp5 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp6 = load float, float* %tmp5, align 4
  %tmp7 = fadd float %tmp6, 2.000000e+00
  store float %tmp7, float* %tmp5, align 4
  br label %bb8

bb8:                                              ; preds = %bb2, %bb4
  %tmp9 = srem i64 %i.0, 3
  %tmp10 = icmp eq i64 %tmp9, 0
  br i1 %tmp10, label %bb15, label %bb11

bb11:                                             ; preds = %bb8
  %tmp12 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp13 = load float, float* %tmp12, align 4
  %tmp14 = fadd float %tmp13, 3.000000e+00
  store float %tmp14, float* %tmp12, align 4
  br label %bb15

bb15:                                             ; preds = %bb8, %bb11
  br label %bb16

bb16:                                             ; preds = %bb15
  %tmp17 = add nsw i64 %i.0, 1
  br label %bb1

bb18:                                             ; preds = %bb1
  ret void
}
