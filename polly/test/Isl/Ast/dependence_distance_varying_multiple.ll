; RUN: opt %loadPolly -basicaa -polly-ast -polly-ast-detect-parallel -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polyhedral-info -polly-check-parallel -analyze < %s | FileCheck %s -check-prefix=PINFO
;
;        void f(int *restrict A, int *restrict B, int *restrict C, int *restrict D,
;               int *restrict E, int N) {
; CHECK:   #pragma minimal dependence distance: N >= 35 ? 1 : N >= 17 && N <= 34 ? 2 : 5
; PINFO:   for.cond: Loop is not parallel.
;          for (int i = 0; i < N; i++) {
;            A[i] = A[100 - 2 * i] + 1;
;            B[i] = B[100 - 3 * i] + 1;
;            C[i] = C[100 - 4 * i] + 1;
;            D[i] = D[100 - 5 * i] + 1;
;            E[i] = E[100 - 6 * i] + 1;
;          }
;        }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* noalias %A, i32* noalias %B, i32* noalias %C, i32* noalias %D, i32* noalias %E, i32 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %mul = shl nsw i32 %i.0, 1
  %sub = sub nsw i32 100, %mul
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %sub
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i32 %i.0
  store i32 %add, i32* %arrayidx1, align 4
  %tmp1 = mul i32 %i.0, -3
  %sub3 = add i32 %tmp1, 100
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i32 %sub3
  %tmp2 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %tmp2, 1
  %arrayidx6 = getelementptr inbounds i32, i32* %B, i32 %i.0
  store i32 %add5, i32* %arrayidx6, align 4
  %mul7 = shl nsw i32 %i.0, 2
  %sub8 = sub nsw i32 100, %mul7
  %arrayidx9 = getelementptr inbounds i32, i32* %C, i32 %sub8
  %tmp3 = load i32, i32* %arrayidx9, align 4
  %add10 = add nsw i32 %tmp3, 1
  %arrayidx11 = getelementptr inbounds i32, i32* %C, i32 %i.0
  store i32 %add10, i32* %arrayidx11, align 4
  %tmp4 = mul i32 %i.0, -5
  %sub13 = add i32 %tmp4, 100
  %arrayidx14 = getelementptr inbounds i32, i32* %D, i32 %sub13
  %tmp5 = load i32, i32* %arrayidx14, align 4
  %add15 = add nsw i32 %tmp5, 1
  %arrayidx16 = getelementptr inbounds i32, i32* %D, i32 %i.0
  store i32 %add15, i32* %arrayidx16, align 4
  %tmp6 = mul i32 %i.0, -6
  %sub18 = add i32 %tmp6, 100
  %arrayidx19 = getelementptr inbounds i32, i32* %E, i32 %sub18
  %tmp7 = load i32, i32* %arrayidx19, align 4
  %add20 = add nsw i32 %tmp7, 1
  %arrayidx21 = getelementptr inbounds i32, i32* %E, i32 %i.0
  store i32 %add20, i32* %arrayidx21, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
