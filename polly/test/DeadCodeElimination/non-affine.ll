; RUN: opt %loadPolly -polly-allow-nonaffine -polly-dce -polly-print-ast -disable-output < %s | FileCheck %s
;
; CHECK: for (int c0 = 0; c0 <= 1023; c0 += 1)
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        A[bar(i)] = i;
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

declare i32 @bar(i32) #1

define void @f(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %nonaff = call i32 @bar(i32 %i.0)
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %nonaff
  store i32 %i.0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

attributes #1 = { nounwind readnone }
