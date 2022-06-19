; RUN: opt %loadPolly -polly-print-dependences -disable-output < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     { Stmt_for_body3[i0, i1] -> Stmt_for_body3[i0 + i1, o1] : i0 >= 0 and 0 <= i1 <= 1023 - i0 and i1 <= 1 and 0 < o1 <= 511 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     { Stmt_for_body3[i0, i1] -> Stmt_for_body3[1 + i0, -1 + i1] : 0 <= i0 <= 1022 and 2 <= i1 <= 511; Stmt_for_body3[i0, 2] -> Stmt_for_body3[2 + i0, 0] : 0 <= i0 <= 1021 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_for_body3[i0, 1] -> Stmt_for_body3[1 + i0, 0] : 0 <= i0 <= 1022 }
;
; void f(int *sum) {
;   for (int i = 0; i < 1024; i++)
;     for (int j = 0; j < 512; j++)
;       sum[i + j] = sum[i] + 3;
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %sum) {
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
  %exitcond = icmp ne i32 %j.0, 512
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %sum, i32 %i.0
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, 3
  %add4 = add nsw i32 %i.0, %j.0
  %arrayidx5 = getelementptr inbounds i32, i32* %sum, i32 %add4
  store i32 %add, i32* %arrayidx5, align 4
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
