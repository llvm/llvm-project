; RUN: opt -S -basic-aa -gvn < %s | FileCheck %s
; RUN: opt -S -basic-aa -gvn -basic-aa-force-at-least-64b=0 < %s | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.6.0"

; The load and store address in the loop body could alias so the load
; can't be hoisted above the store and out of the loop.

declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1)

define i32 @foo(i32 %x, i32 %z, i32 %n) {
entry:
  %pool = alloca [59 x i32], align 4
  %tmp = bitcast [59 x i32]* %pool to i8*
  call void @llvm.memset.p0i8.i32(i8* align 4 nonnull %tmp, i8 0, i32 236, i1 false)
  %cmp3 = icmp eq i32 %n, 0
  br i1 %cmp3, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add = add i32 %z, %x
  %and = and i32 %add, 2147483647
  %sub = add nsw i32 %and, -2137521902
  %arrayidx = getelementptr inbounds [59 x i32], [59 x i32]* %pool, i32 0, i32 %sub
  %arrayidx1 = getelementptr inbounds [59 x i32], [59 x i32]* %pool, i32 0, i32 42
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  store i32 %i.04, i32* %arrayidx, align 4
  %tmp1 = load i32, i32* %arrayidx1, align 4
  %inc = add nuw i32 %i.04, 1
  %exitcond = icmp ne i32 %inc, %n
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  %lcssa = phi i32 [ %tmp1, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %s = phi i32 [ 0, %entry ], [ %lcssa, %for.end.loopexit ]
; CHECK: ret i32 %s
  ret i32 %s
}
