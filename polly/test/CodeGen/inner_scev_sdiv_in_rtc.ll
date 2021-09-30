; RUN: opt %loadPolly -polly-codegen \
; RUN:     -S < %s | FileCheck %s
;
; This will just check that we generate valid code here.
;
; CHECK: polly.start:
;
;    void f(int *A, int *B) {
;      for (int i = 0; i < 1024; i++)
;        A[i % 3] = B[i / 42];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32* %B, i32 %N) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb9, %bb
  %i.0 = phi i32 [ 0, %bb ], [ %tmp10, %bb9 ]
  %exitcond = icmp ne i32 %i.0, %N
  br i1 %exitcond, label %bb2, label %bb11

bb2:                                              ; preds = %bb1
  %tmp = sdiv i32 %i.0, 42
  %tmp3 = sext i32 %tmp to i64
  %tmp4 = getelementptr inbounds i32, i32* %B, i64 %tmp3
  %tmp5 = load i32, i32* %tmp4, align 4
  %tmp6 = srem i32 %i.0, 3
  %tmp7 = sext i32 %tmp6 to i64
  %tmp8 = getelementptr inbounds i32, i32* %A, i64 %tmp7
  store i32 %tmp5, i32* %tmp8, align 4
  br label %bb9

bb9:                                              ; preds = %bb2
  %tmp10 = add nuw nsw i32 %i.0, 1
  br label %bb1

bb11:                                             ; preds = %bb1
  ret void
}
