; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-invariant-load-hoisting=true -polly-process-unprofitable -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-invariant-load-hoisting=true -polly-process-unprofitable -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Invariant Accesses:
; CHECK-NEXT: ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:   { Stmt_bb1[i0] -> MemRef_UB[0] };
;
;    void f(int *A, int *UB) {
;      for (int i = 0; i < *UB; i++)
;        A[i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, ptr %UB) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb6 ], [ 0, %bb ]
  %tmp = load i32, ptr %UB, align 4
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = icmp slt i64 %indvars.iv, %tmp2
  br i1 %tmp3, label %bb4, label %bb7

bb4:                                              ; preds = %bb1
  %tmp5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 0, ptr %tmp5, align 4
  br label %bb6

bb6:                                              ; preds = %bb4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}
