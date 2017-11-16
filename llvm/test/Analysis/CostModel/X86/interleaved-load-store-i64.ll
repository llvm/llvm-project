; REQUIRES: asserts
; RUN: opt -S -loop-vectorize -debug-only=loop-vectorize -mcpu=core-avx2 %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@i64src = common local_unnamed_addr global [120 x i64] zeroinitializer, align 4
@i64dst = common local_unnamed_addr global [120 x i64] zeroinitializer, align 4

; Function Attrs: norecurse nounwind
define void @stride2i64(i64 %k, i32 %width_) {
entry:

; CHECK: Found an estimated cost of 8 for VF 4 For instruction:   %0 = load i64
; CHECK: Found an estimated cost of 8 for VF 4 For instruction:   store i64

  %cmp27 = icmp sgt i32 %width_, 0
  br i1 %cmp27, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:
  %i.028 = phi i32 [ 0, %for.body.lr.ph ], [ %add16, %for.body ]
  %arrayidx = getelementptr inbounds [120 x i64], [120 x i64]* @i64src, i32 0, i32 %i.028
  %0 = load i64, i64* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [120 x i64], [120 x i64]* @i64dst, i32 0, i32 %i.028
  store i64 %0, i64* %arrayidx2, align 4
  %add4 = add nuw nsw i32 %i.028, 1
  %arrayidx5 = getelementptr inbounds [120 x i64], [120 x i64]* @i64src, i32 0, i32 %add4
  %1 = load i64, i64* %arrayidx5, align 4
  %arrayidx8 = getelementptr inbounds [120 x i64], [120 x i64]* @i64dst, i32 0, i32 %add4
  store i64 %1, i64* %arrayidx8, align 4
  %add16 = add nuw nsw i32 %i.028, 2
  %cmp = icmp slt i32 %add16, %width_
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

