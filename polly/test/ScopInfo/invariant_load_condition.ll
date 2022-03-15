; RUN: opt %loadPolly -polly-process-unprofitable -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
;
; CHECK: Invariant Accesses:
; CHECK-NEXT: ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:  { Stmt_bb2[i0] -> MemRef_C[0] };
;
;    void f(int *A, int *C) {
;      for (int i = 0; i < 1024; i++)
;        if (*C)
;          A[i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32* %C) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = load i32, i32* %C, align 4
  %tmp3 = icmp eq i32 %tmp, 0
  br i1 %tmp3, label %bb6, label %bb4

bb4:                                              ; preds = %bb2
  %tmp5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 0, i32* %tmp5, align 4
  br label %bb6

bb6:                                              ; preds = %bb2, %bb4
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
