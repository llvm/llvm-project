; RUN: opt %loadPolly -polly-dependences -analyze < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     { Stmt_S2[i0, 0] -> Stmt_S1[1 + i0, 0] : i0 <= 97 and i0 >= 0; Stmt_S1[i0, 0] -> Stmt_S2[i0, 0] : i0 <= 98 and i0 >= 0 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     { Stmt_S2[i0, 0] -> Stmt_S1[1 + i0, 0] : i0 <= 97 and i0 >= 0; Stmt_S1[i0, 0] -> Stmt_S2[i0, 0] : i0 <= 98 and i0 >= 0 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_S2[i0, i1] -> Stmt_S2[1 + i0, i1] : i0 <= 97 and i0 >= 0 and i1 <= 99 and i1 >= 1 }
;
;    void f(int *sum) {
;      for (int i = 0; i < 99; i++) {
;        for (int j = 0; j < 1; j++)
; S1:      sum[j] += 42;
;        for (int j = 0; j < 100; j++)
; S2:      sum[j] += i * j;
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %sum)  {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc12, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc13, %for.inc12 ]
  %exitcond2 = icmp ne i32 %i.0, 99
  br i1 %exitcond2, label %for.body, label %for.end14

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 1
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  br label %S1

S1:                                               ; preds = %for.body3
  %arrayidx = getelementptr inbounds i32, i32* %sum, i32 %j.0
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, 42
  store i32 %add, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %S1
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc9, %for.end
  %j.1 = phi i32 [ 0, %for.end ], [ %inc10, %for.inc9 ]
  %exitcond1 = icmp ne i32 %j.1, 100
  br i1 %exitcond1, label %for.body6, label %for.end11

for.body6:                                        ; preds = %for.cond4
  br label %S2

S2:                                               ; preds = %for.body6
  %mul = mul nsw i32 %i.0, %j.1
  %arrayidx7 = getelementptr inbounds i32, i32* %sum, i32 %j.1
  %tmp3 = load i32, i32* %arrayidx7, align 4
  %add8 = add nsw i32 %tmp3, %mul
  store i32 %add8, i32* %arrayidx7, align 4
  br label %for.inc9

for.inc9:                                         ; preds = %S2
  %inc10 = add nsw i32 %j.1, 1
  br label %for.cond4

for.end11:                                        ; preds = %for.cond4
  br label %for.inc12

for.inc12:                                        ; preds = %for.end11
  %inc13 = add nsw i32 %i.0, 1
  br label %for.cond

for.end14:                                        ; preds = %for.cond
  ret void
}
