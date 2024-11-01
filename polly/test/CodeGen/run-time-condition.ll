; RUN: opt %loadPolly -basic-aa -polly-codegen -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @run-time-condition(ptr noalias %A, ptr noalias %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %indvar = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i64 %indvar, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %load = load i16, ptr %B
  %add10 = add nsw i16 %load, 1
  %arrayidx13 = getelementptr inbounds i16, ptr %A, i64 %indvar
  store i16 %add10, ptr %arrayidx13, align 2
  %inc = add nsw i64 %indvar, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; The trivial case, no run-time checks required.
;
; CHECK: polly.split_new_and_old:
; CHECK:  br i1 true, label %polly.start, label %for.cond
