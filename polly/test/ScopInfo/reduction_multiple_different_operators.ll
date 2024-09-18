; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; Should not be identified as reduction as there are different operations
; involved on sum (multiplication followed by addition)
; CHECK: Reduction Type: NONE
;
;    void f(int *restrict sum) {
;      for (int i = 0; i < 1024; i++) {
;        *sum = (*sum * 5) + 25;
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* noalias %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = load i32, i32* %sum, align 4
  %tmp1 = mul i32 %tmp, 5
  %mul = add i32 %tmp1, 25
  store i32 %mul, i32* %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
