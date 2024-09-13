; RUN: opt %loadPolly -basic-aa -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: Reduction Type: +
; CHECK: MemRef_sum
; CHECK: Reduction Type: NONE
; CHECK: MemRef_A
; CHECK: Reduction Type: +
; CHECK: MemRef_sum
; CHECK-NOT: MemRef_A
;
;    void f(int *restrict sum, int *restrict A) {
;      for (int i = 0; i < 1024; i++)
;        *sum = (A[i + 3] * (i - 14)) + ((A[i] + *sum + A[0]) + A[1023]) +
;               (A[i + 2] * A[i - 1]);
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* noalias %sum, i32* noalias %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %i.0, 3
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %add
  %tmp = load i32, i32* %arrayidx, align 4
  %sub = add nsw i32 %i.0, -14
  %mul = mul nsw i32 %tmp, %sub
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i32 %i.0
  %tmp1 = load i32, i32* %arrayidx1, align 4
  %tmp2 = load i32, i32* %sum, align 4
  %add2 = add nsw i32 %tmp1, %tmp2
  %tmp3 = load i32, i32* %A, align 4
  %add4 = add nsw i32 %add2, %tmp3
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i32 1023
  %tmp4 = load i32, i32* %arrayidx5, align 4
  %add6 = add nsw i32 %add4, %tmp4
  %add7 = add nsw i32 %mul, %add6
  %add8 = add nsw i32 %i.0, 2
  %arrayidx9 = getelementptr inbounds i32, i32* %A, i32 %add8
  %tmp5 = load i32, i32* %arrayidx9, align 4
  %sub10 = add nsw i32 %i.0, -1
  %arrayidx11 = getelementptr inbounds i32, i32* %A, i32 %sub10
  %tmp6 = load i32, i32* %arrayidx11, align 4
  %mul12 = mul nsw i32 %tmp5, %tmp6
  %add13 = add nsw i32 %add7, %mul12
  store i32 %add13, i32* %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
