; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: Reduction Type: +
; CHECK: Reduction Type: +
; CHECK: Reduction Type: *
; CHECK: Reduction Type: *
;
; void f(int *sums) {
;   for (int i = 0; i < 1024; i++) {
;     for (int j = 0; j < 1024; j++) {
;       sums[i] += 5;
;       sums[i+1024] *= 5;
;     }
;   }
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %sums) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  %exitcond1 = icmp ne i32 %i.0, 1024
  br i1 %exitcond1, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %sums, i32 %i.0
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, 5
  store i32 %add, i32* %arrayidx, align 4
  %add4 = add nsw i32 %i.0, 1024
  %arrayidx5 = getelementptr inbounds i32, i32* %sums, i32 %add4
  %tmp2 = load i32, i32* %arrayidx5, align 4
  %mul = mul nsw i32 %tmp2, 5
  store i32 %mul, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %inc7 = add nsw i32 %i.0, 1
  br label %for.cond

for.end8:                                         ; preds = %for.cond
  ret void
}
