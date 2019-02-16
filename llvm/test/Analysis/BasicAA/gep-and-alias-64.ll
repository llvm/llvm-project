; RUN: opt -S -basicaa -gvn < %s | FileCheck %s

target datalayout = "e-m:o-p:64:64-f64:32:64-f80:128-n8:16:32-S128"
target triple = "x86_64-apple-macosx10.6.0"

; The load and store address in the loop body could alias so the load
; can't be hoisted above the store and out of the loop.

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1)

define i64 @foo(i64 %x, i64 %z, i64 %n) {
entry:
  %pool = alloca [59 x i64], align 4
  %tmp = bitcast [59 x i64]* %pool to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %tmp, i8 0, i64 236, i32 4, i1 false)
  %cmp3 = icmp eq i64 %n, 0
  br i1 %cmp3, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add = add i64 %z, %x
  %and = and i64 %add, 9223372036854775807
  %sub = add nsw i64 %and, -9223372036844814062
  %arrayidx = getelementptr inbounds [59 x i64], [59 x i64]* %pool, i64 0, i64 %sub
  %arrayidx1 = getelementptr inbounds [59 x i64], [59 x i64]* %pool, i64 0, i64 42
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  store i64 %i.04, i64* %arrayidx, align 4
  %tmp1 = load i64, i64* %arrayidx1, align 4
  %inc = add nuw i64 %i.04, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  %lcssa = phi i64 [ %tmp1, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %s = phi i64 [ 0, %entry ], [ %lcssa, %for.end.loopexit ]
; CHECK: ret i64 %s
  ret i64 %s
}
