; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void f(char *A, char N, char p) {
;      for (char i = 0; i < N; i++)
;        A[p-1] = 0;
;    }

; CHECK:      Function: wrap
;
; CHECK:      Context:
; CHECK-NEXT: [N, p] -> {  : -128 <= N <= 127 and -128 <= p <= 127 }
;
; CHECK:      Invalid Context:
; CHECK-NEXT: [N, p] -> {  : p = -128 and N > 0 }

target datalayout = "e-m:e-i8:64-f80:128-n8:16:32:64-S128"

define void @wrap(i8* %A, i8 %N, i8 %p) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i8 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %tmp3 = icmp slt i8 %indvars.iv, %N
  br i1 %tmp3, label %bb4, label %bb8

bb4:                                              ; preds = %bb2
  %tmp5 = add i8 %p, -1
  %tmp6 = getelementptr i8, i8* %A, i8 %tmp5
  store i8 0, i8* %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb4
  %indvars.iv.next = add nuw nsw i8 %indvars.iv, 1
  br label %bb2

bb8:                                              ; preds = %bb2
  ret void
}
