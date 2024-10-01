; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Verify that we can look through a bitcast when delinearizing multi-dimensional
; arrays.

; CHECK: Stmt_bb7[i0, i1] -> MemRef_B[i0, i1]
; CHECK: Stmt_bb7[i0, i1] -> MemRef_B[i0, i1]
; CHECK: Stmt_bb17[i0] -> MemRef_B[i0, 100]

define void @kernel(ptr %A, ptr %B, ptr %C, ptr %D) {
bb:
  br label %bb4

bb4:                                              ; preds = %bb21, %bb
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %bb21 ], [ 0, %bb ]
  %exitcond3 = icmp eq i64 %indvars.iv1, 100
  br i1 %exitcond3, label %bb22, label %bb5

bb5:                                              ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb16, %bb5
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb16 ], [ 0, %bb5 ]
  %exitcond = icmp eq i64 %indvars.iv, 100
  br i1 %exitcond, label %bb17, label %bb7

bb7:                                              ; preds = %bb6
  %tmp = getelementptr inbounds float, ptr %D, i64 %indvars.iv
  %tmp8 = load float, ptr %tmp, align 4
  %tmp9 = getelementptr inbounds [101 x float], ptr %B, i64 %indvars.iv1, i64 %indvars.iv
  %tmp10 = load float, ptr %tmp9, align 4
  %tmp11 = fmul float %tmp8, %tmp10
  %tmp12 = getelementptr inbounds [101 x float], ptr %C, i64 %indvars.iv1, i64 %indvars.iv
  store float %tmp11, ptr %tmp12, align 4
  %tmp13 = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %tmp141 = load i32, ptr %tmp13, align 4
  %tmp15 = getelementptr inbounds [101 x float], ptr %B, i64 %indvars.iv1, i64 %indvars.iv
  store i32 %tmp141, ptr %tmp15, align 4
  br label %bb16

bb16:                                             ; preds = %bb7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb6

bb17:                                             ; preds = %bb6
  %tmp18 = trunc i64 %indvars.iv1 to i32
  %tmp19 = sitofp i32 %tmp18 to float
  %tmp20 = getelementptr inbounds [101 x float], ptr %B, i64 %indvars.iv1, i64 100
  store float %tmp19, ptr %tmp20, align 4
  br label %bb21

bb21:                                             ; preds = %bb17
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %bb4

bb22:                                             ; preds = %bb4
  ret void
}
