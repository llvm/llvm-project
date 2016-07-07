; RUN: opt -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @ff()

define void @test(i8* noalias nocapture %base, i64 %size) #1 {
entry:
  %cmp3 = icmp eq i64 %size, 0
  br i1 %cmp3, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
; CHECK-LABEL: @test(
; CHECK-NOT: llvm.memset
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  tail call void @ff()
  %arrayidx = getelementptr inbounds i8, i8* %base, i64 %indvars.iv
  store i8 0, i8* %arrayidx, align 1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %size
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

attributes #1 = { uwtable }
