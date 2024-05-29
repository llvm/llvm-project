; RUN: opt %loadNPMPolly -polly-allow-nonaffine-branches '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        if (A[i])
;          if (A[i - 1])
;            A[i] = A[i - 2];
;    }
;
; CHECK:    Region: %bb1---%bb18
; CHECK:    Max Loop Depth:  1
; CHECK:    Statements {
; CHECK:      Stmt_bb2__TO__bb16
; CHECK:            Schedule :=
; CHECK:                { Stmt_bb2__TO__bb16[i0] -> [i0] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2__TO__bb16[i0] -> MemRef_A[i0] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2__TO__bb16[i0] -> MemRef_A[-1 + i0] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2__TO__bb16[i0] -> MemRef_A[-2 + i0] };
; CHECK:            MayWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2__TO__bb16[i0] -> MemRef_A[i0] };
; CHECK:    }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb17, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb17 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb18

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp3 = load i32, ptr %tmp, align 4
  %tmp4 = icmp eq i32 %tmp3, 0
  br i1 %tmp4, label %bb16, label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nsw i64 %indvars.iv, -1
  %tmp7 = getelementptr inbounds i32, ptr %A, i64 %tmp6
  %tmp8 = load i32, ptr %tmp7, align 4
  %tmp9 = icmp eq i32 %tmp8, 0
  br i1 %tmp9, label %bb15, label %bb10

bb10:                                             ; preds = %bb5
  %tmp11 = add nsw i64 %indvars.iv, -2
  %tmp12 = getelementptr inbounds i32, ptr %A, i64 %tmp11
  %tmp13 = load i32, ptr %tmp12, align 4
  %tmp14 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %tmp13, ptr %tmp14, align 4
  br label %bb15

bb15:                                             ; preds = %bb5, %bb10
  br label %bb16

bb16:                                             ; preds = %bb2, %bb15
  br label %bb17

bb17:                                             ; preds = %bb16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb18:                                             ; preds = %bb1
  ret void
}
