; RUN: opt %loadPolly -polly-dependences -analyze < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     { Stmt_S1[i0, i1] -> Stmt_S2[i0] : 0 <= i0 <= 99 and 0 <= i1 <= 99; Stmt_S0[i0] -> Stmt_S1[i0, o1] : 0 <= i0 <= 99 and 0 <= o1 <= 99; Stmt_S2[i0] -> Stmt_S0[1 + i0] : 0 <= i0 <= 98 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     { Stmt_S1[i0, i1] -> Stmt_S2[i0] : 0 <= i0 <= 99 and 0 <= i1 <= 99; Stmt_S0[i0] -> Stmt_S1[i0, o1] : 0 <= i0 <= 99 and 0 <= o1 <= 99; Stmt_S2[i0] -> Stmt_S0[1 + i0] : 0 <= i0 <= 98 }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     { Stmt_S1[i0, i1] -> Stmt_S2[i0] : 0 <= i0 <= 99 and 0 <= i1 <= 99; Stmt_S0[i0] -> Stmt_S1[i0, o1] : 0 <= i0 <= 99 and 0 <= o1 <= 99; Stmt_S2[i0] -> Stmt_S0[1 + i0] : 0 <= i0 <= 98 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_S1[i0, i1] -> Stmt_S1[i0, 1 + i1] : 0 <= i0 <= 99 and 0 <= i1 <= 98 }
;
;    void f(int *sum) {
;      for (int i = 0; i < 100; i++) {
; S0:    *sum *= 42;
;        for (int j = 0; j < 100; j++)
; S1:      *sum += i * j;
; S2:    *sum *= 7;
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %sum)  {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  %exitcond1 = icmp ne i32 %i.0, 100
  br i1 %exitcond1, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  br label %S0

S0:                                               ; preds = %for.body
  %tmp = load i32, i32* %sum, align 4
  %mul = mul nsw i32 %tmp, 42
  store i32 %mul, i32* %sum, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %S0
  %j.0 = phi i32 [ 0, %S0 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 100
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  br label %S1

S1:                                               ; preds = %for.body3
  %mul4 = mul nsw i32 %i.0, %j.0
  %tmp2 = load i32, i32* %sum, align 4
  %add = add nsw i32 %tmp2, %mul4
  store i32 %add, i32* %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %S1
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %S2

S2:                                               ; preds = %for.end
  %tmp3 = load i32, i32* %sum, align 4
  %mul5 = mul nsw i32 %tmp3, 7
  store i32 %mul5, i32* %sum, align 4
  br label %for.inc6

for.inc6:                                         ; preds = %S2
  %inc7 = add nsw i32 %i.0, 1
  br label %for.cond

for.end8:                                         ; preds = %for.cond
  ret void
}
