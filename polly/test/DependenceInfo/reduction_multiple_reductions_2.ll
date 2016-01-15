; RUN: opt %loadPolly -basicaa -polly-dependences -analyze < %s | FileCheck %s
;
;
; These are the important RAW dependences, as they need to originate/end in only one iteration:
;   Stmt_S1[i0, 1023] -> Stmt_S2[i0, o1]
;   Stmt_S1[i0, i1] -> Stmt_S2[i0, 0]
;
; These are the important WAW dependences, as they need to originate/end in only one iteration:
;   Stmt_S1[i0, 1023] -> Stmt_S2[i0, o1]
;   Stmt_S1[i0, i1] -> Stmt_S2[i0, 0]
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     { Stmt_S0[i0] -> Stmt_S1[i0, o1] : i0 <= 1023 and i0 >= 0 and o1 <= 1023 and o1 >= 0; Stmt_S2[i0, i1] -> Stmt_S3[i0] : i0 <= 1023 and i0 >= 0 and i1 <= 1023 and i1 >= 0; Stmt_S3[i0] -> Stmt_S0[1 + i0] : i0 <= 1022 and i0 >= 0; Stmt_S1[i0, 1023] -> Stmt_S2[i0, o1] : i0 <= 1023 and i0 >= 0 and o1 <= 1023 and o1 >= 0; Stmt_S1[i0, i1] -> Stmt_S2[i0, 0] : i0 <= 1023 and i0 >= 0 and i1 <= 1022 and i1 >= 0 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     {  }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     { Stmt_S0[i0] -> Stmt_S1[i0, o1] : i0 <= 1023 and i0 >= 0 and o1 <= 1023 and o1 >= 0; Stmt_S2[i0, i1] -> Stmt_S3[i0] : i0 <= 1023 and i0 >= 0 and i1 <= 1023 and i1 >= 0; Stmt_S3[i0] -> Stmt_S0[1 + i0] : i0 <= 1022 and i0 >= 0; Stmt_S1[i0, 1023] -> Stmt_S2[i0, o1] : i0 <= 1023 and i0 >= 0 and o1 <= 1023 and o1 >= 0; Stmt_S1[i0, i1] -> Stmt_S2[i0, 0] : i0 <= 1023 and i0 >= 0 and i1 <= 1022 and i1 >= 0 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_S1[i0, i1] -> Stmt_S1[i0, 1 + i1] : i0 <= 1023 and i0 >= 0 and i1 <= 1022 and i1 >= 0; Stmt_S2[i0, i1] -> Stmt_S2[i0, 1 + i1] : i0 <= 1023 and i0 >= 0 and i1 <= 1022 and i1 >= 0 }
;
;    void f(int *restrict red) {
;      for (int j = 0; j < 1024; j++) {
; S0:    *red = 42 + *red * 5;
;        for (int i = 0; i < 1024; i++)
; S1:      *red *= i;
;        for (int i = 0; i < 1024; i++)
; S2:      *red += i;
; S3:    *red = 42 + *red * 7;
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* noalias %red)  {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc15, %entry
  %j.0 = phi i32 [ 0, %entry ], [ %inc16, %for.inc15 ]
  %exitcond2 = icmp ne i32 %j.0, 1024
  br i1 %exitcond2, label %for.body, label %for.end17

for.body:                                         ; preds = %for.cond
  br label %S0

S0:                                               ; preds = %for.body
  %tmp = load i32, i32* %red, align 4
  %mul = mul nsw i32 %tmp, 5
  %add = add nsw i32 %mul, 42
  store i32 %add, i32* %red, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %S0
  %i.0 = phi i32 [ 0, %S0 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  br label %S1

S1:                                               ; preds = %for.body3
  %tmp3 = load i32, i32* %red, align 4
  %mul4 = mul nsw i32 %tmp3, %i.0
  store i32 %mul4, i32* %red, align 4
  br label %for.inc

for.inc:                                          ; preds = %S1
  %inc = add nsw i32 %i.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc10, %for.end
  %i5.0 = phi i32 [ 0, %for.end ], [ %inc11, %for.inc10 ]
  %exitcond1 = icmp ne i32 %i5.0, 1024
  br i1 %exitcond1, label %for.body8, label %for.end12

for.body8:                                        ; preds = %for.cond6
  br label %S2

S2:                                               ; preds = %for.body8
  %tmp4 = load i32, i32* %red, align 4
  %add9 = add nsw i32 %tmp4, %i5.0
  store i32 %add9, i32* %red, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %S2
  %inc11 = add nsw i32 %i5.0, 1
  br label %for.cond6

for.end12:                                        ; preds = %for.cond6
  br label %S3

S3:                                               ; preds = %for.end12
  %tmp5 = load i32, i32* %red, align 4
  %mul13 = mul nsw i32 %tmp5, 7
  %add14 = add nsw i32 %mul13, 42
  store i32 %add14, i32* %red, align 4
  br label %for.inc15

for.inc15:                                        ; preds = %S3
  %inc16 = add nsw i32 %j.0, 1
  br label %for.cond

for.end17:                                        ; preds = %for.cond
  ret void
}
