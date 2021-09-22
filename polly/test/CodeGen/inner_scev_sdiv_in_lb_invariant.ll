; RUN: opt %loadPolly -S -polly-codegen \
; RUN:     < %s | FileCheck %s
;
; Check that this will not crash our code generation.
;
; CHECK: polly.start:
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N / 4; i++)
;        A[i] += A[i - 1];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N) {
bb:
  %tmp = sdiv i32 %N, 4
  %tmp2 = sext i32 %tmp to i64
  br label %bb1

bb1:                                              ; preds = %bb11, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb11 ], [ 0, %bb ]
  %tmp3 = icmp slt i64 %indvars.iv, %tmp2
  br i1 %tmp3, label %bb4, label %bb12

bb4:                                              ; preds = %bb1
  %tmp5 = add nsw i64 %indvars.iv, -1
  %tmp6 = getelementptr inbounds i32, i32* %A, i64 %tmp5
  %tmp7 = load i32, i32* %tmp6, align 4
  %tmp8 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp9 = load i32, i32* %tmp8, align 4
  %tmp10 = add nsw i32 %tmp9, %tmp7
  store i32 %tmp10, i32* %tmp8, align 4
  br label %bb11

bb11:                                             ; preds = %bb4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb12:                                             ; preds = %bb1
  ret void
}
