; RUN: opt %loadPolly -polly-ast -polly-ast-detect-parallel -analyze < %s | FileCheck %s
;
; CHECK: #pragma known-parallel reduction (^ : MemRef_sum)
;        void f(int N, int M, int *sum) {
;          for (int i = 0; i < N; i++)
; CHECK:    #pragma simd
;            for (int j = 0; j < M; j++)
;              sum[j] ^= j;
;        }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32 %N, i32 %M, i32* %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc4, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc5, %for.inc4 ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end6

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i32 %j.0, %M
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %sum, i32 %j.0
  %tmp = load i32, i32* %arrayidx, align 4
  %xor = xor i32 %tmp, %j.0
  store i32 %xor, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc4

for.inc4:                                         ; preds = %for.end
  %inc5 = add nsw i32 %i.0, 1
  br label %for.cond

for.end6:                                         ; preds = %for.cond
  ret void
}
