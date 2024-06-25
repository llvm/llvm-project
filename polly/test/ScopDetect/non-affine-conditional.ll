; RUN: opt %loadNPMPolly -polly-allow-nonaffine-branches '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        if (A[i])
;          A[i] = 0;
;    }
;
; CHECK: Valid Region for Scop: bb1 => bb9
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb8, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb8 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb9

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp3 = load i32, ptr %tmp, align 4
  %tmp4 = icmp eq i32 %tmp3, 0
  br i1 %tmp4, label %bb7, label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 0, ptr %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb2, %bb5
  br label %bb8

bb8:                                              ; preds = %bb7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb9:                                              ; preds = %bb1
  ret void
}
