; RUN: opt %loadPolly -basicaa -polly-stmt-granularity=bb -polly-scops -analyze < %s | FileCheck %s
;
; void f(int N, int * restrict sums, int * restrict escape) {
;   int i, j;
;   for (i = 0; i < 1024; i++) {
;     for (j = 0; j < 1024; j++) {
;       sums[i] += 5;
;       escape[N-j] = escape[i] + sums[i-1];
;     }
;   }
; }
;
; CHECK: Reduction Type: NONE
; CHECK: sums
; CHECK: Reduction Type: NONE
; CHECK: sums
; CHECK: Reduction Type: NONE
; CHECK: escape
; CHECK: Reduction Type: NONE
; CHECK: sums
; CHECK: Reduction Type: NONE
; CHECK: escape
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32 %N, i32* noalias %sums, i32* noalias %escape) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc11, %for.inc10 ]
  %exitcond1 = icmp ne i32 %i.0, 1024
  br i1 %exitcond1, label %for.body, label %for.end12

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
  %arrayidx4 = getelementptr inbounds i32, i32* %escape, i32 %i.0
  %tmp2 = load i32, i32* %arrayidx4, align 4
  %sub = add nsw i32 %i.0, -1
  %arrayidx5 = getelementptr inbounds i32, i32* %sums, i32 %sub
  %tmp3 = load i32, i32* %arrayidx5, align 4
  %add6 = add nsw i32 %tmp2, %tmp3
  %sub7 = sub nsw i32 %N, %i.0
  %add8 = add nsw i32 %sub7, %j.0
  %arrayidx9 = getelementptr inbounds i32, i32* %escape, i32 %add8
  store i32 %add6, i32* %arrayidx9, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %inc11 = add nsw i32 %i.0, 1
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  ret void
}
