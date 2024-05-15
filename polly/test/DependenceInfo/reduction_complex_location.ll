; RUN: opt %loadPolly -polly-print-dependences -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-print-dependences -polly-dependences-analysis-level=reference-wise -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-print-dependences -polly-dependences-analysis-level=access-wise -disable-output < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_for_body3[i0, i1] -> Stmt_for_body3[2 + i0, -1 + i1] : 0 <= i0 <= 97 and 0 < i1 <= 99 }
;
; void f(int *sum) {
;   for (int i = 0; i < 100; i++)
;     for (int j = 0; j < 100; j++)
;       sum[i+j*2] += j * i;
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  %exitcond1 = icmp ne i32 %i.0, 100
  br i1 %exitcond1, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 100
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %mul = mul nsw i32 %j.0, %i.0
  %mul4 = shl nsw i32 %j.0, 1
  %add = add nsw i32 %i.0, %mul4
  %arrayidx = getelementptr inbounds i32, ptr %sum, i32 %add
  %tmp = load i32, ptr %arrayidx, align 4
  %add5 = add nsw i32 %tmp, %mul
  store i32 %add5, ptr %arrayidx, align 4
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
