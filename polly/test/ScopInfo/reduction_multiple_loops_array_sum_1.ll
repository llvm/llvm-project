; RUN: opt -basic-aa %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: Stmt_for_body
; CHECK: Reduction Type: NONE
; CHECK: MemRef_sum_04
; CHECK: Reduction Type: NONE
; CHECK: MemRef_sum_12
; CHECK: Stmt_for_inc
; CHECK: Reduction Type: +
; CHECK: MemRef_sum_12
; CHECK: Reduction Type: NONE
; CHECK: MemRef_A
; CHECK: Reduction Type: +
; CHECK: MemRef_sum_12
; CHECK: Stmt_for_inc5
; CHECK: Reduction Type: NONE
; CHECK: MemRef_sum_12
; CHECK: Reduction Type: NONE
; CHECK: MemRef_sum_04
;
; int f(int * __restrict__ A) {
;   int i, j, sum = 1;
;   for (i = 0; i < 100; i++) {
;     sum *= 7;
;     for (j = 0; j < 100; j++) {
;       sum += A[i+j];
;     }
;   }
;   return sum;
; }
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @f(ptr noalias %A) {
entry:
  %sum.04.reg2mem = alloca i32
  %sum.12.reg2mem = alloca i32
  br label %entry.split

entry.split:                                      ; preds = %entry
  store i32 0, ptr %sum.04.reg2mem
  br label %for.body

for.body:                                         ; preds = %for.inc5, %entry.split
  %indvars.iv23 = phi i64 [ 0, %entry.split ], [ %3, %for.inc5 ]
  %sum.04.reload = load i32, ptr %sum.04.reg2mem
  %mul = mul nsw i32 %sum.04.reload, 7
  store i32 %mul, ptr %sum.12.reg2mem
  br label %for.inc

for.inc:                                          ; preds = %for.inc, %for.body
  %indvars.iv1 = phi i64 [ 0, %for.body ], [ %1, %for.inc ]
  %sum.12.reload = load i32, ptr %sum.12.reg2mem
  %0 = add i64 %indvars.iv23, %indvars.iv1
  %arrayidx = getelementptr i32, ptr %A, i64 %0
  %tmp5 = load i32, ptr %arrayidx, align 4
  %add4 = add nsw i32 %tmp5, %sum.12.reload
  %1 = add nuw nsw i64 %indvars.iv1, 1
  %exitcond1 = icmp eq i64 %1, 100
  store i32 %add4, ptr %sum.12.reg2mem
  br i1 %exitcond1, label %for.inc5, label %for.inc

for.inc5:                                         ; preds = %for.inc
  %2 = load i32, ptr %sum.12.reg2mem
  %3 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond2 = icmp eq i64 %3, 100
  store i32 %2, ptr %sum.04.reg2mem
  br i1 %exitcond2, label %for.end7, label %for.body

for.end7:                                         ; preds = %for.inc5
  %4 = load i32, ptr %sum.04.reg2mem
  ret i32 %4
}
