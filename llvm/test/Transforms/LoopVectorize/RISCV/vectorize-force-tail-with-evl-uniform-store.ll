; RUN: opt < %s --prefer-predicate-over-epilogue=predicate-dont-vectorize --passes=loop-vectorize -mcpu=sifive-p470 -mattr=+v,+f
; RUN: opt < %s --prefer-predicate-over-epilogue=predicate-dont-vectorize --passes=loop-vectorize -mcpu=sifive-p470 -mattr=+v,+f -force-tail-folding-style=data-with-evl
; Generated from issue #109468.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

define void @lshift_significand(i32 %n, ptr nocapture writeonly %0) local_unnamed_addr #0 {
entry:
  %cmp1.peel = icmp eq i32 %n, 0
  %spec.select = select i1 %cmp1.peel, i64 2, i64 0
  br label %for.body9

for.body9:                                        ; preds = %entry, %for.body9
  %indvars.iv = phi i64 [ %spec.select, %entry ], [ %indvars.iv.next, %for.body9 ]
  %1 = sub nuw nsw i64 1, %indvars.iv
  %arrayidx13 = getelementptr [3 x i64], ptr %0, i64 0, i64 %1
  store i64 0, ptr %arrayidx13, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 3
  br i1 %exitcond.not, label %for.end16, label %for.body9

for.end16:                                        ; preds = %for.body9
  ret void
}
