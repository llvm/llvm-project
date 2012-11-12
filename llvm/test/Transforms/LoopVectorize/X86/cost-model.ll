; RUN: opt < %s  -loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@c = common global [2048 x i32] zeroinitializer, align 16
@b = common global [2048 x i32] zeroinitializer, align 16
@d = common global [2048 x i32] zeroinitializer, align 16
@a = common global [2048 x i32] zeroinitializer, align 16

;CHECK: cost_model_1
;CHECK: <4 x i32>
;CHECK: ret void
define void @cost_model_1() nounwind uwtable noinline ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds [2048 x i32]* @c, i64 0, i64 %0
  %1 = load i32* %arrayidx, align 8
  %idxprom1 = sext i32 %1 to i64
  %arrayidx2 = getelementptr inbounds [2048 x i32]* @b, i64 0, i64 %idxprom1
  %2 = load i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds [2048 x i32]* @d, i64 0, i64 %indvars.iv
  %3 = load i32* %arrayidx4, align 4
  %idxprom5 = sext i32 %3 to i64
  %arrayidx6 = getelementptr inbounds [2048 x i32]* @a, i64 0, i64 %idxprom5
  store i32 %2, i32* %arrayidx6, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
