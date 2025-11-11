; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
;    void f(int *A, int N, int p) {
;      for (int i = 0; i < N; i++)
;        A[i + 30] = 0;
;    }
;
; The wrap function has no inbounds GEP but the nowrap function has. Therefore,
; we will add the assumption that i+1 won't overflow only to the former.
;
; Note: 2147483618 + 30 == 2 ^ 31

; CHECK:      Function: wrap
;
; CHECK:      Context:
; CHECK-NEXT: [N] -> {  : -2147483648 <= N <= 2147483647 }
;
; CHECK:      Invalid Context:
; CHECK-NEXT: [N] -> {  : N >= 2147483619 }

target datalayout = "e-m:e-i32:64-f80:128-n8:16:32:64-S128"

define void @wrap(ptr %A, i32 %N, i32 %p) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i32 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %tmp3 = icmp slt i32 %indvars.iv, %N
  br i1 %tmp3, label %bb4, label %bb8

bb4:                                              ; preds = %bb2
  %tmp5 = add i32 %indvars.iv, 30
  %tmp6 = getelementptr i32, ptr %A, i32 %tmp5
  store i32 0, ptr %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb4
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  br label %bb2

bb8:                                              ; preds = %bb2
  ret void
}
