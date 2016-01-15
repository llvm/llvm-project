; RUN: opt %loadPolly -polly-dependences -analyze -basicaa < %s | FileCheck %s
;
; CHECK:      Reduction dependences:
; CHECK-NEXT:     { Stmt_for_inc[i0, i1] -> Stmt_for_inc[i0, 1 + i1] : 0 <= i0 <= 99 and 0 <= i1 <= 98 }
;
; int f(int * __restrict__ A) {
;   int i, j, sum = 0;
;   for (k = 0; k < 37; k = g(k)) {
;     for (i = 0; i < 100; i++) {
;       sum *= 2;
;       for (j = 0; j < 100; j++) {
;         sum += A[i+j];
;       }
;     }
;   }
;   return sum;
; }
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @g(i64)

define i32 @f(i32* noalias %A) {
entry:
  %sum.04.reg2mem = alloca i32
  %sum.12.reg2mem = alloca i32
  br label %entry.split

entry.split:                                      ; preds = %entry
  store i32 0, i32* %sum.04.reg2mem
  br label %for.body_outer_split

for.body_outer_split:                             ; preds = %entry.split, %for.inc5
  %indvars.ivK = phi i64 [ 0, %entry.split ], [ %incK, %for.bos2 ]
  br label %for.body_outer

for.body_outer:                                   ; preds = %for.body_outer_split
  %incK = call i64 @g(i64 %indvars.ivK)
  %exitcondK = icmp eq i64 %incK, 100
  br i1 %exitcondK, label %for.end7, label %for.body

for.body:                                         ; preds = %for.inc5, %for.body_outer
  %indvars.iv23 = phi i64 [ 0, %for.body_outer ], [ %3, %for.inc5 ]
  %sum.04.reload = load i32, i32* %sum.04.reg2mem
  %mul = shl nsw i32 %sum.04.reload, 1
  store i32 %mul, i32* %sum.12.reg2mem
  br label %for.inc

for.inc:                                          ; preds = %for.inc, %for.body
  %indvars.iv1 = phi i64 [ 0, %for.body ], [ %1, %for.inc ]
  %sum.12.reload = load i32, i32* %sum.12.reg2mem
  %0 = add i64 %indvars.iv23, %indvars.iv1
  %arrayidx = getelementptr i32, i32* %A, i64 %0
  %tmp5 = load i32, i32* %arrayidx, align 4
  %add4 = add nsw i32 %tmp5, %sum.12.reload
  %1 = add nuw nsw i64 %indvars.iv1, 1
  %exitcond1 = icmp eq i64 %1, 100
  store i32 %add4, i32* %sum.12.reg2mem
  br i1 %exitcond1, label %for.inc5, label %for.inc

for.inc5:                                         ; preds = %for.inc
  %2 = load i32, i32* %sum.12.reg2mem
  %3 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond2 = icmp eq i64 %3, 100
  store i32 %2, i32* %sum.04.reg2mem
  br i1 %exitcond2, label %for.bos2, label %for.body

for.bos2:
  br label %for.body_outer_split

for.end7:                                         ; preds = %for.inc5
  %4 = load i32, i32* %sum.04.reg2mem
  ret i32 %4
}
