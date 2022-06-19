; RUN: opt %loadPolly -polly-codegen -polly-vectorizer=polly < %s

; This test case checks that the polly vectorizer does not crash when
; calculating the number of iterations.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@b = external global [2048 x i64], align 16

define void @foo(i64 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %indvar = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i64 %indvar, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds [2048 x i64], [2048 x i64]* @b, i64 0, i64 %indvar
  store i64 1, i64* %arrayidx
  %inc = add nsw i64 %indvar, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

