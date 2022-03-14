; RUN: opt -basic-aa %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: Stmt_for_body
; CHECK: Reduction Type: *
; CHECK: MemRef_sum
; CHECK: Reduction Type: *
; CHECK: MemRef_sum
; CHECK: Stmt_for_body3
; CHECK: Reduction Type: NONE
; CHECK: MemRef_A
; CHECK: Reduction Type: +
; CHECK: MemRef_sum
; CHECK: Reduction Type: +
; CHECK: MemRef_sum
; CHECK: Stmt_for_end
; CHECK: Reduction Type: *
; CHECK: MemRef_sum
; CHECK: Reduction Type: *
; CHECK: MemRef_sum
;
; void f(int *restrict A, int *restrict sum) {
;   int i, j;
;   for (i = 0; i < 100; i++) {
;     *sum *= 7;
;     for (j = 0; j < 100; j++) {
;       *sum += A[i + j];
;     }
;     *sum *= 5;
;   }
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* noalias %A, i32* noalias %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  %exitcond1 = icmp ne i32 %i.0, 100
  br i1 %exitcond1, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  %tmp = load i32, i32* %sum, align 4
  %mul = mul nsw i32 %tmp, 7
  store i32 %mul, i32* %sum, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 100
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %add = add nsw i32 %i.0, %j.0
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %add
  %tmp2 = load i32, i32* %arrayidx, align 4
  %tmp3 = load i32, i32* %sum, align 4
  %add4 = add nsw i32 %tmp3, %tmp2
  store i32 %add4, i32* %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %tmp4 = load i32, i32* %sum, align 4
  %mul5 = mul nsw i32 %tmp4, 5
  store i32 %mul5, i32* %sum, align 4
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %inc7 = add nsw i32 %i.0, 1
  br label %for.cond

for.end8:                                         ; preds = %for.cond
  ret void
}
