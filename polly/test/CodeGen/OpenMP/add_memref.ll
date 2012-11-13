; RUN: opt %loadPolly %defaultOpts -polly-codegen -enable-polly-openmp -disable-verify -S < %s | FileCheck %s

;#define N 10
;
;void foo() {
;  float A[N];
;
;  for (int i=0; i < N; i++)
;    A[i] = 10;
;
;  return;
;}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

define void @foo() nounwind {
entry:
  %A = alloca [10 x float], align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds [10 x float]* %A, i32 0, i32 %i.0
  store float 1.000000e+01, float* %arrayidx
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK: store [10 x float]* %A, [10 x float]**
