; RUN: opt %loadPolly -polly-dependences -analyze < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     [N] -> { Stmt_S0[] -> Stmt_S1[o0] : N >= 11 and 0 <= o0 <= 1023; Stmt_S1[i0] -> Stmt_S2[] : N >= 11 and 0 <= i0 <= 1023 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     [N] -> { Stmt_S1[i0] -> Stmt_S2[] : N >= 11 and 0 <= i0 <= 1023 }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     [N] -> { Stmt_S0[] -> Stmt_S1[o0] : N >= 11 and 0 <= o0 <= 1023; Stmt_S1[i0] -> Stmt_S2[] : N >= 11 and 0 <= i0 <= 1023 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     [N] -> { Stmt_S1[i0] -> Stmt_S1[1 + i0] : N >= 11 and 0 <= i0 <= 1022 }
;
;    void f(int *sum, int N) {
;      if (N >= 10) {
; S0:    *sum = 0;
;        for (int i = 0; i < 1024; i++)
; S1:      *sum += i;
; S2:    *sum = *sum * 3;
;      }
;    }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %sum, i32 %N) {
entry:
  br label %entry.1

entry.1:
  %excond = icmp sgt i32 %N, 10
  br i1 %excond, label %S0, label %f.end

S0:
  store i32 0, i32* %sum, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %S0
  %i.0 = phi i32 [ 0, %S0 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %S1, label %S2

S1:                                               ; preds = %for.cond
  %tmp = load i32, i32* %sum, align 4
  %add = add nsw i32 %tmp, %i.0
  store i32 %add, i32* %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %S1
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

S2:                                               ; preds = %for.cond
  %tmp1 = load i32, i32* %sum, align 4
  %mul = mul nsw i32 %tmp1, 3
  store i32 %mul, i32* %sum, align 4
  br label %f.end

f.end:
  ret void
}
