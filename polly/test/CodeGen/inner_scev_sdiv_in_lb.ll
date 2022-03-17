; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s --check-prefix=CODEGEN
;
; CHECK: [N] -> { Stmt_bb11[i0, i1] : i0 < N and i1 >= 0 and 3i1 <= -3 + i0 };
; CODEGEN: polly
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++)
;        for (int j = 0; j < i / 3; j++)
;          A[i] += A[j];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N) {
bb:
  %tmp = sext i32 %N to i64
  br label %bb3

bb3:                                              ; preds = %bb19, %bb
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %bb19 ], [ 0, %bb ]
  %tmp4 = icmp slt i64 %indvars.iv1, %tmp
  br i1 %tmp4, label %bb5, label %bb20

bb5:                                              ; preds = %bb3
  br label %bb6

bb6:                                              ; preds = %bb17, %bb5
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb17 ], [ 0, %bb5 ]
  %tmp7 = trunc i64 %indvars.iv1 to i32
  %tmp8 = sdiv i32 %tmp7, 3
  %tmp9 = sext i32 %tmp8 to i64
  %tmp10 = icmp slt i64 %indvars.iv, %tmp9
  br i1 %tmp10, label %bb11, label %bb18

bb11:                                             ; preds = %bb6
  %tmp12 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp13 = load i32, i32* %tmp12, align 4
  %tmp14 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv1
  %tmp15 = load i32, i32* %tmp14, align 4
  %tmp16 = add nsw i32 %tmp15, %tmp13
  store i32 %tmp16, i32* %tmp14, align 4
  br label %bb17

bb17:                                             ; preds = %bb11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb6

bb18:                                             ; preds = %bb6
  br label %bb19

bb19:                                             ; preds = %bb18
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %bb3

bb20:                                             ; preds = %bb3
  ret void
}
