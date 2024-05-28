; RUN: opt %loadNPMPolly '-passes=print<polly-dependences>' -disable-output < %s | FileCheck %s
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     { Stmt_S1[i0] -> Stmt_S3[2 + i0] : 0 <= i0 <= 96; Stmt_S2[i0, i1] -> Stmt_S3[o0] : i1 <= 1 - i0 and -i1 < o0 <= 1 and o0 <= 1 + i0 - i1; Stmt_S3[i0] -> Stmt_S2[o0, 1 - i0] : 0 <= i0 <= 1 and i0 < o0 <= 98 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     { Stmt_S1[i0] -> Stmt_S3[2 + i0] : 0 <= i0 <= 96; Stmt_S2[i0, i1] -> Stmt_S3[o0] : i1 <= 1 - i0 and -i1 < o0 <= 1 and o0 <= 1 + i0 - i1; Stmt_S3[i0] -> Stmt_S2[o0, 1 - i0] : 0 <= i0 <= 1 and i0 < o0 <= 98 }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     { Stmt_S1[i0] -> Stmt_S3[2 + i0] : 0 <= i0 <= 96; Stmt_S2[i0, i1] -> Stmt_S3[o0] : i1 <= 1 - i0 and -i1 < o0 <= 1 and o0 <= 1 + i0 - i1; Stmt_S3[i0] -> Stmt_S2[o0, 1 - i0] : 0 <= i0 <= 1 and i0 < o0 <= 98 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_S2[i0, i1] -> Stmt_S2[1 + i0, i1] : 0 <= i0 <= 97 and i1 >= 0 and 2 - i0 <= i1 <= 98 - i0; Stmt_S2[0, 0] -> Stmt_S2[1, 0] }
;
;    void f(int *sum) {
;      int i, j;
;      for (i = 0; i < 99; i++) {
; S1:    sum[i + 1] += 42;
;        for (j = i; j < 100; j++)
; S2:      sum[i - j] += i * j;
; S3:    sum[i - 1] += 7;
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr %sum)  {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc11, %for.inc10 ]
  %exitcond1 = icmp ne i32 %i.0, 99
  br i1 %exitcond1, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  br label %S1

S1:                                               ; preds = %for.body
  %add = add nsw i32 %i.0, 1
  %arrayidx = getelementptr inbounds i32, ptr %sum, i32 %add
  %tmp = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %tmp, 42
  store i32 %add1, ptr %arrayidx, align 4
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc, %S1
  %j.0 = phi i32 [ %i.0, %S1 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 100
  br i1 %exitcond, label %for.body4, label %for.end

for.body4:                                        ; preds = %for.cond2
  br label %S2

S2:                                               ; preds = %for.body4
  %mul = mul nsw i32 %i.0, %j.0
  %sub = sub nsw i32 %i.0, %j.0
  %arrayidx5 = getelementptr inbounds i32, ptr %sum, i32 %sub
  %tmp2 = load i32, ptr %arrayidx5, align 4
  %add6 = add nsw i32 %tmp2, %mul
  store i32 %add6, ptr %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %S2
  %inc = add nsw i32 %j.0, 1
  br label %for.cond2

for.end:                                          ; preds = %for.cond2
  br label %S3

S3:                                               ; preds = %for.end
  %sub7 = add nsw i32 %i.0, -1
  %arrayidx8 = getelementptr inbounds i32, ptr %sum, i32 %sub7
  %tmp3 = load i32, ptr %arrayidx8, align 4
  %add9 = add nsw i32 %tmp3, 7
  store i32 %add9, ptr %arrayidx8, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %S3
  %inc11 = add nsw i32 %i.0, 1
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  ret void
}
