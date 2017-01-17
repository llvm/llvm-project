; RUN: opt -S -inline < %s | FileCheck %s
; RUN: opt -S -passes='cgscc(inline)' < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define void @Body(i32* nocapture %res, i32* nocapture readnone %c, i32* nocapture readonly %d, i32* nocapture readonly %p, i32 %i) #0 {
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %cmp = icmp eq i32 %0, 0
  %arrayidx2 = getelementptr inbounds i32, i32* %res, i64 %idxprom
  %1 = load i32, i32* %arrayidx2, align 4
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %arrayidx6 = getelementptr inbounds i32, i32* %d, i64 %idxprom
  %2 = load i32, i32* %arrayidx6, align 4
  %add = add nsw i32 %2, %1
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.false
  %cond = phi i32 [ %add, %cond.false ], [ %1, %entry ]
  store i32 %cond, i32* %arrayidx2, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @Test(i32* %res, i32* %c, i32* %d, i32* %p, i32 %n) #1 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 1600
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @Body(i32* %res, i32* undef, i32* %d, i32* %p, i32 %i.0), !llvm.mem.parallel_loop_access !0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !0

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK-LABEL: @Test
; CHECK: load i32,{{.*}}, !llvm.mem.parallel_loop_access !0
; CHECK: load i32,{{.*}}, !llvm.mem.parallel_loop_access !0
; CHECK: load i32,{{.*}}, !llvm.mem.parallel_loop_access !0
; CHECK: store i32{{.*}}, !llvm.mem.parallel_loop_access !0
; CHECK: br label %for.cond, !llvm.loop !0

attributes #0 = { norecurse nounwind uwtable }

!0 = distinct !{!0}

