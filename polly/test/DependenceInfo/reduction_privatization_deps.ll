; RUN: opt %loadPolly -polly-dependences -analyze < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     { Stmt_S0[i0] -> Stmt_S1[o0, i0 - o0] : i0 <= 1023 and 0 <= o0 <= i0; Stmt_S1[i0, i1] -> Stmt_S2[-1 + i0 + i1] : 0 <= i0 <= 1023 and i1 >= 0 and -i0 < i1 <= 1024 - i0 and i1 <= 1023 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     { Stmt_S2[i0] -> Stmt_S2[1 + i0] : 0 <= i0 <= 1022; Stmt_S1[0, 0] -> Stmt_S2[0] }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     { Stmt_S0[i0] -> Stmt_S1[o0, i0 - o0] : i0 <= 1023 and 0 <= o0 <= i0; Stmt_S1[i0, i1] -> Stmt_S2[i0 + i1] : i0 >= 0 and 0 <= i1 <= 1023 - i0 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_S1[i0, i1] -> Stmt_S1[1 + i0, -1 + i1] : 0 <= i0 <= 1022 and 0 < i1 <= 1023 }
;
;    void f(int *sum) {
;      for (int i = 0; i < 1024; i++)
; S0:    sum[i] = 0;
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 1024; j++)
; S1:      sum[i + j] += i;
;      for (int i = 0; i < 1024; i++)
; S2:    sum[i] = sum[i + 1] * 3;
;    }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %sum)  {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond3 = icmp ne i32 %i.0, 1024
  br i1 %exitcond3, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %S0

S0:                                               ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %sum, i32 %i.0
  store i32 0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %S0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc13, %for.end
  %i1.0 = phi i32 [ 0, %for.end ], [ %inc14, %for.inc13 ]
  %exitcond2 = icmp ne i32 %i1.0, 1024
  br i1 %exitcond2, label %for.body4, label %for.end15

for.body4:                                        ; preds = %for.cond2
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc10, %for.body4
  %j.0 = phi i32 [ 0, %for.body4 ], [ %inc11, %for.inc10 ]
  %exitcond1 = icmp ne i32 %j.0, 1024
  br i1 %exitcond1, label %for.body7, label %for.end12

for.body7:                                        ; preds = %for.cond5
  br label %S1

S1:                                               ; preds = %for.body7
  %add = add nsw i32 %i1.0, %j.0
  %arrayidx8 = getelementptr inbounds i32, i32* %sum, i32 %add
  %tmp = load i32, i32* %arrayidx8, align 4
  %add9 = add nsw i32 %tmp, %i1.0
  store i32 %add9, i32* %arrayidx8, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %S1
  %inc11 = add nsw i32 %j.0, 1
  br label %for.cond5

for.end12:                                        ; preds = %for.cond5
  br label %for.inc13

for.inc13:                                        ; preds = %for.end12
  %inc14 = add nsw i32 %i1.0, 1
  br label %for.cond2

for.end15:                                        ; preds = %for.cond2
  br label %for.cond17

for.cond17:                                       ; preds = %for.inc23, %for.end15
  %i16.0 = phi i32 [ 0, %for.end15 ], [ %inc24, %for.inc23 ]
  %exitcond = icmp ne i32 %i16.0, 1024
  br i1 %exitcond, label %for.body19, label %for.end25

for.body19:                                       ; preds = %for.cond17
  br label %S2

S2:                                               ; preds = %for.body19
  %add20 = add nsw i32 %i16.0, 1
  %arrayidx21 = getelementptr inbounds i32, i32* %sum, i32 %add20
  %tmp4 = load i32, i32* %arrayidx21, align 4
  %mul = mul nsw i32 %tmp4, 3
  %arrayidx22 = getelementptr inbounds i32, i32* %sum, i32 %i16.0
  store i32 %mul, i32* %arrayidx22, align 4
  br label %for.inc23

for.inc23:                                        ; preds = %S2
  %inc24 = add nsw i32 %i16.0, 1
  br label %for.cond17

for.end25:                                        ; preds = %for.cond17
  ret void
}
