; RUN: opt < %s -passes='dse,loop-vectorize' -S | FileCheck %s
;
; Verify that LoopVectorize does not crash when BlockFrequencyInfo is computed
; after an earlier loop has been vectorized and a prior pass (DSE) has cached
; CycleAnalysis.
;
; CHECK-LABEL: @InitializeMasks(

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@clear_mask = external global [65 x i64]

define void @InitializeMasks(ptr %mask_1) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr [8 x i8], ptr @clear_mask, i64 %indvars.iv
  store i64 1, ptr %arrayidx, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  store i64 0, ptr %mask_1, align 8
  br label %for.body98

for.body98:
  %indvars.iv3551 = phi i64 [ 0, %for.end ], [ %indvars.iv.next356, %for.inc140 ]
  %indvars.iv.next356 = add i64 %indvars.iv3551, 1
  %cmp107 = icmp ugt i64 %indvars.iv3551, 15
  br i1 %cmp107, label %if.end, label %if.end.thread

if.end.thread:
  store i64 0, ptr %mask_1, align 8
  br label %for.inc140

if.end:
  br i1 false, label %if.end.if.then128_crit_edge, label %for.inc140

if.end.if.then128_crit_edge:
  ret void

for.inc140:
  %exitcond358.not = icmp eq i64 %indvars.iv3551, 56
  br i1 %exitcond358.not, label %for.cond180.loopexit.7, label %for.body98

for.cond180.loopexit.7:
  ret void
}
