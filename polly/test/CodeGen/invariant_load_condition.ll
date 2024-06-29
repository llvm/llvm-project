; RUN: opt %loadNPMPolly -polly-process-unprofitable -passes=polly-codegen -polly-invariant-load-hoisting=true -S < %s | FileCheck %s
;
; CHECK-LABEL: polly.preload.begin:
; CHECK-NEXT:     %polly.access.C = getelementptr i32, ptr %C, i64 0
; CHECK-NEXT:     %polly.access.C.load = load i32, ptr %polly.access.C
; CHECK-NOT:      %polly.access.C.load = load i32, ptr %polly.access.C
;
; CHECK-LABEL: polly.cond:
; CHECK-NEXT:   %[[R0:[0-9]*]] = sext i32 %polly.access.C.load to i64
; CHECK-NEXT:   %[[R1:[0-9]*]] = icmp sle i64 %[[R0]], -1
; CHECK-NEXT:   %[[R2:[0-9]*]] = sext i32 %polly.access.C.load to i64
; CHECK-NEXT:   %[[R3:[0-9]*]] = icmp sge i64 %[[R2]], 1
; CHECK-NEXT:   %[[R4:[0-9]*]] = or i1 %[[R1]], %[[R3]]
; CHECK-NEXT:   br i1 %[[R4]]
;
; CHECK-NOT:  polly.stmt.bb2
;
;    void f(int *A, int *C) {
;      for (int i = 0; i < 1024; i++)
;        if (*C)
;          A[i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, ptr %C) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = load i32, ptr %C, align 4
  %tmp3 = icmp eq i32 %tmp, 0
  br i1 %tmp3, label %bb6, label %bb4

bb4:                                              ; preds = %bb2
  %tmp5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 0, ptr %tmp5, align 4
  br label %bb6

bb6:                                              ; preds = %bb2, %bb4
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
