; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -enable-if-conversion -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @reduction_func(
;CHECK-NOT: load <4 x i32>
;CHECK: ret i32
define i32 @reduction_func(i32* nocapture %A, i32 %n) nounwind uwtable readonly ssp {
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %sum.011 = phi i32 [ %sum.1, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 30
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %add = add i32 %sum.011, 2
  %add4 = add i32 %add, %0
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %sum.1 = phi i32 [ %add4, %if.then ], [ %sum.011, %for.body ]
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ 4, %for.inc ]
  ret i32 %sum.0.lcssa
}

