; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(int *A, int N, int p) {
;      for (int i = 0; i < N; i++)
;        A[i + p] = 0;
;    }
;
; Note: 2147483648 == 2 ^ 31
;
; CHECK:      Function: wrap
; CHECK:      Invalid Context:
; CHECK:      [N, p] -> {  : p >= 2147483649 - N }
;
target datalayout = "e-m:e-i32:64-f80:128-n8:16:32:64-S128"

define void @wrap(ptr %A, i32 %N, i32 %p) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i32 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %tmp3 = icmp slt i32 %indvars.iv, %N
  br i1 %tmp3, label %bb4, label %bb8

bb4:                                              ; preds = %bb2
  %tmp5 = add i32 %indvars.iv, %p
  %tmp6 = getelementptr inbounds i32, ptr %A, i32 %tmp5
  store i32 0, ptr %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb4
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  br label %bb2

bb8:                                              ; preds = %bb2
  ret void
}
