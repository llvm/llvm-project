; RUN: opt -basic-aa %loadPolly -polly-dependences -analyze < %s | FileCheck %s
; RUN: opt -basic-aa %loadPolly -polly-dependences -polly-dependences-analysis-level=reference-wise -analyze < %s | FileCheck %s
; RUN: opt -basic-aa %loadPolly -polly-dependences -polly-dependences-analysis-level=access-wise -analyze < %s | FileCheck %s
;
; Verify that only the inner reduction like accesses cause reduction dependences
;
; CHECK:      Reduction dependences:
; CHECK-NEXT:     { Stmt_for_body3[i0, i1] -> Stmt_for_body3[i0, 1 + i1] : 0 <= i0 <= 99 and 0 <= i1 <= 98 }
;
; void f(int * restrict A, int * restrict sum) {
;   int i, j, k;
;   for (i = 0; i < 100; i++) {
;     *sum *= 7;
;     for (j = 0; j < 100; j++) {
;       *sum += A[i+j];
;       for (k = 0; k< 100; k++) {}
;     }
;   }
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* noalias %A, i32* noalias %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc11, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc12, %for.inc11 ]
  %exitcond2 = icmp ne i32 %i.0, 100
  br i1 %exitcond2, label %for.body, label %for.end13

for.body:                                         ; preds = %for.cond
  %tmp = load i32, i32* %sum, align 4
  %mul = mul nsw i32 %tmp, 7
  store i32 %mul, i32* %sum, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc8, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc9, %for.inc8 ]
  %exitcond1 = icmp ne i32 %j.0, 100
  br i1 %exitcond1, label %for.body3, label %for.end10

for.body3:                                        ; preds = %for.cond1
  %add = add nsw i32 %i.0, %j.0
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %add
  %tmp3 = load i32, i32* %arrayidx, align 4
  %tmp4 = load i32, i32* %sum, align 4
  %add4 = add nsw i32 %tmp4, %tmp3
  store i32 %add4, i32* %sum, align 4
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i32 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %k.0, 100
  br i1 %exitcond, label %for.body7, label %for.end

for.body7:                                        ; preds = %for.cond5
  br label %for.inc

for.inc:                                          ; preds = %for.body7
  %inc = add nsw i32 %k.0, 1
  br label %for.cond5

for.end:                                          ; preds = %for.cond5
  br label %for.inc8

for.inc8:                                         ; preds = %for.end
  %inc9 = add nsw i32 %j.0, 1
  br label %for.cond1

for.end10:                                        ; preds = %for.cond1
  br label %for.inc11

for.inc11:                                        ; preds = %for.end10
  %inc12 = add nsw i32 %i.0, 1
  br label %for.cond

for.end13:                                        ; preds = %for.cond
  ret void
}
