; RUN: opt %loadPolly -basicaa -polly-import-jscop -polly-codegen -polly-vectorizer=polly -S < %s | FileCheck %s
;
; Check that we use the correct __new__ strides:
;    stride zero for B
;    stride one for A
;
; CHECK:  %polly.access.B = getelementptr i32, i32* %B, i64 0
; CHECK:  %[[BC:[._a-zA-Z0-9]*]] = bitcast i32* %polly.access.B to <1 x i32>*
; CHECK:  %[[LD:[._a-zA-Z0-9]*]] = load <1 x i32>, <1 x i32>* %[[BC]], align 8
; CHECK:  %[[SV:[._a-zA-Z0-9]*]] = shufflevector <1 x i32> %[[LD]], <1 x i32> %[[LD]], <16 x i32> zeroinitializer
;
; CHECK:  %polly.access.A = getelementptr i32, i32* %A, i64 0
; CHECK:  %[[VP:[._a-zA-Z0-9]*]] = bitcast i32* %polly.access.A to <16 x i32>*
; CHECK:  store <16 x i32> %[[SV]], <16 x i32>* %[[VP]], align 8
;
;    void simple_stride(int *restrict A, int *restrict B) {
;      for (int i = 0; i < 16; i++)
;        A[i * 2] = B[i * 2];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @simple_stride(i32* noalias %A, i32* noalias %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 16
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = shl nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %tmp
  %tmp4 = load i32, i32* %arrayidx, align 4
  %tmp5 = shl nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %tmp5
  store i32 %tmp4, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
