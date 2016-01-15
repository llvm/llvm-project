; RUN: opt %loadPolly -polly-dependences -analyze -basicaa < %s | FileCheck %s
;
; CHECK:      Reduction dependences:
; CHECK-NEXT:     [N] -> { Stmt_for_body3[i0, i1] -> Stmt_for_body3[i0, 1 + i1] : i0 <= 1023 and i0 >= 0 and i1 <= 1022 and i1 >= 0 and i1 >= 1024 - N + i0 }
;
; void f(int N, int * restrict sums, int * restrict escape) {
;   for (int i = 0; i < 1024; i++) {
;     for (int j = 0; j < 1024; j++) {
;       sums[i] += 5;
;       if (N - i + j < 1024)
;         escape[N - i + j] = sums[i];
;     }
;   }
; }
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
  %sub = sub nsw i32 %N, %i.0
  %add4 = add nsw i32 %sub, %j.0
  %cmp5 = icmp slt i32 %add4, 1024
  br i1 %cmp5, label %if.then, label %if.end

if.then:                                          ; preds = %for.body3
  %arrayidx6 = getelementptr inbounds i32, i32* %sums, i32 %i.0
  %tmp2 = load i32, i32* %arrayidx6, align 4
  %sub7 = sub nsw i32 %N, %i.0
  %add8 = add nsw i32 %sub7, %j.0
  %arrayidx9 = getelementptr inbounds i32, i32* %escape, i32 %add8
  store i32 %tmp2, i32* %arrayidx9, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body3
  br label %for.inc

for.inc:                                          ; preds = %if.end
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
