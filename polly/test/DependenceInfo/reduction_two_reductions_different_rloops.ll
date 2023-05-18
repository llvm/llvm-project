; RUN: opt %loadPolly -basic-aa -polly-stmt-granularity=bb -polly-print-dependences -disable-output < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_for_body3[i0, i1] -> Stmt_for_body3[o0, 1 + i0 + i1 - o0] : i0 >= 0 and i1 >= 0 and o0 >= -1022 + i0 + i1 and i0 <= o0 <= 1023 and o0 <= 1 + i0 }
;
; void f(int *restrict A, int *restrict B, int *restrict Values) {
;   for (int i = 0; i < 1024; i++) {
;     for (int j = 0; j < 1024; j++) {
;       A[i] += Values[i + j - 1];
;       B[j] += Values[i + j + 42];
;     }
;   }
; }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr noalias %A, ptr noalias %B, ptr noalias %Values)  {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc11, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc12, %for.inc11 ]
  %exitcond1 = icmp ne i32 %i.0, 1024
  br i1 %exitcond1, label %for.body, label %for.end13

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %add = add nsw i32 %i.0, %j.0
  %sub = add nsw i32 %add, -1
  %arrayidx = getelementptr inbounds i32, ptr %Values, i32 %sub
  %tmp = load i32, ptr %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i32 %i.0
  %tmp2 = load i32, ptr %arrayidx4, align 4
  %add5 = add nsw i32 %tmp2, %tmp
  store i32 %add5, ptr %arrayidx4, align 4
  %add6 = add nsw i32 %i.0, %j.0
  %add7 = add nsw i32 %add6, 42
  %arrayidx8 = getelementptr inbounds i32, ptr %Values, i32 %add7
  %tmp3 = load i32, ptr %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32, ptr %B, i32 %j.0
  %tmp4 = load i32, ptr %arrayidx9, align 4
  %add10 = add nsw i32 %tmp4, %tmp3
  store i32 %add10, ptr %arrayidx9, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc11

for.inc11:                                        ; preds = %for.end
  %inc12 = add nsw i32 %i.0, 1
  br label %for.cond

for.end13:                                        ; preds = %for.cond
  ret void
}
