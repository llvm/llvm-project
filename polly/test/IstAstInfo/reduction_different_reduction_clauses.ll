; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa '-passes=print<polly-ast>' -polly-ast-detect-parallel -disable-output < %s | FileCheck %s
;
; CHECK: #pragma simd reduction (+ : MemRef_sum{{[1,2]}}, MemRef_sum{{[1,2]}}) reduction (* : MemRef_prod) reduction (| : MemRef_or) reduction (& : MemRef_and)
; CHECK: #pragma known-parallel reduction (+ : MemRef_sum{{[1,2]}}, MemRef_sum{{[1,2]}}) reduction (* : MemRef_prod) reduction (| : MemRef_or) reduction (& : MemRef_and)
; CHECK: for (int c0 = 0; c0 < N; c0 += 1)
; CHECK:   Stmt_for_body(c0);
;
;    void f(int N, int *restrict sum1, int *restrict sum2, int *restrict prod,
;           int *restrict and, int *restrict or ) {
;      for (int i = 0; i < N; i++) {
;        *sum1 += i;
;        *sum2 += i + 1;
;        *prod *= i;
;        *and &= i;
;        * or |= i;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32 %N, ptr noalias %sum1, ptr noalias %sum2, ptr noalias %prod, ptr noalias %and, ptr noalias %or) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = load i32, ptr %sum1, align 4
  %add = add nsw i32 %tmp, %i.0
  store i32 %add, ptr %sum1, align 4
  %add1 = add nsw i32 %i.0, 1
  %tmp1 = load i32, ptr %sum2, align 4
  %add2 = add nsw i32 %tmp1, %add1
  store i32 %add2, ptr %sum2, align 4
  %tmp2 = load i32, ptr %prod, align 4
  %mul = mul nsw i32 %tmp2, %i.0
  store i32 %mul, ptr %prod, align 4
  %tmp3 = load i32, ptr %and, align 4
  %and3 = and i32 %tmp3, %i.0
  store i32 %and3, ptr %and, align 4
  %tmp4 = load i32, ptr %or, align 4
  %or4 = or i32 %tmp4, %i.0
  store i32 %or4, ptr %or, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
