; RUN: opt %loadPolly -polly-allow-nonaffine -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(int *A) {
;      for (int i = 0; i < 128; i++)
;        for (int j = 0; j < 16; j++)
;          A[i * j]++;
;    }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb7
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb7[i0, i1] : 0 <= i0 <= 127 and 0 <= i1 <= 15 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> [i0, i1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_A[o0] : 0 <= o0 <= 2048 };
; CHECK-NEXT:         MayWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_A[o0] : 0 <= o0 <= 2048 };
; CHECK-NEXT: }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
bb:
  br label %bb4

bb4:                                              ; preds = %bb13, %bb
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %bb13 ], [ 0, %bb ]
  %exitcond3 = icmp ne i64 %indvars.iv1, 128
  br i1 %exitcond3, label %bb5, label %bb14

bb5:                                              ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb11, %bb5
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb11 ], [ 0, %bb5 ]
  %exitcond = icmp ne i64 %indvars.iv, 16
  br i1 %exitcond, label %bb7, label %bb12

bb7:                                              ; preds = %bb6
  %tmp = mul nsw i64 %indvars.iv1, %indvars.iv
  %tmp8 = getelementptr inbounds i32, i32* %A, i64 %tmp
  %tmp9 = load i32, i32* %tmp8, align 4
  %tmp10 = add nsw i32 %tmp9, 1
  store i32 %tmp10, i32* %tmp8, align 4
  br label %bb11

bb11:                                             ; preds = %bb7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb6

bb12:                                             ; preds = %bb6
  br label %bb13

bb13:                                             ; preds = %bb12
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %bb4

bb14:                                             ; preds = %bb4
  ret void
}
