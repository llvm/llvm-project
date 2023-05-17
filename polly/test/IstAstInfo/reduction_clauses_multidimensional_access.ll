; RUN: opt %loadPolly -polly-print-ast -polly-ast-detect-parallel -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -print-polyhedral-info -polly-check-parallel -disable-output < %s | FileCheck %s -check-prefix=PINFO
;
; CHECK: #pragma known-parallel reduction (^ : MemRef_sum)
;        void f(int N, int M, int P, int sum[P][M]) {
; PINFO:   for.cond: Loop is not parallel.
;          for (int i = 0; i < N; i++)
; PINFO-NEXT: for.cond1: Loop is parallel.
;             for (int j = 0; j < P; j++)
; CHECK:        #pragma simd
; PINFO-NEXT:   for.cond4: Loop is parallel.
;               for (int k = 0; k < M; k++)
;                 sum[j][k] ^= j;
;        }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32 %N, i32 %M, i32 %P, ptr %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc11, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc12, %for.inc11 ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end13

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc8, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc9, %for.inc8 ]
  %cmp2 = icmp slt i32 %j.0, %P
  br i1 %cmp2, label %for.body3, label %for.end10

for.body3:                                        ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i32 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %cmp5 = icmp slt i32 %k.0, %M
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %tmp = mul nsw i32 %j.0, %M
  %arrayidx.sum = add i32 %tmp, %k.0
  %arrayidx7 = getelementptr inbounds i32, ptr %sum, i32 %arrayidx.sum
  %tmp1 = load i32, ptr %arrayidx7, align 4
  %xor = xor i32 %tmp1, %j.0
  store i32 %xor, ptr %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %inc = add nsw i32 %k.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
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
