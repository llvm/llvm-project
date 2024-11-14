; RUN: opt %loadPolly -basic-aa -polly-print-scops -disable-output < %s | FileCheck %s
;
; void f(int N, int * restrict sums, int * restrict escape) {
;   int i, j;
;   for (i = 0; i < 1024; i++) {
;     sums[i] += 5;
;     escape[i] = sums[i];
;   }
; }
;
; CHECK: Reduction Type: NONE
; CHECK: sums
; CHECK: Reduction Type: NONE
; CHECK: sums
; CHECK: Reduction Type: NONE
; CHECK: escape
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32 %N, i32* noalias %sums, i32* noalias %escape) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc8, %for.inc ]
  %exitcond1 = icmp ne i32 %i.0, 1024
  br i1 %exitcond1, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %sums, i32 0
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, 5
  store i32 %add, i32* %arrayidx, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %escape, i32 %i.0
  store i32 %add, i32* %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc8 = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
