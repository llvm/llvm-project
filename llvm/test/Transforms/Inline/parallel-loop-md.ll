; RUN: opt -S -passes=inline < %s | FileCheck %s
; RUN: opt -S -passes='cgscc(inline)' < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define void @Body(ptr nocapture %res, ptr nocapture readnone %c, ptr nocapture readonly %d, ptr nocapture readonly %p, i32 %i) #0 {
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, ptr %p, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %cmp = icmp eq i32 %0, 0
  %arrayidx2 = getelementptr inbounds i32, ptr %res, i64 %idxprom
  %1 = load i32, ptr %arrayidx2, align 4
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %arrayidx6 = getelementptr inbounds i32, ptr %d, i64 %idxprom
  %2 = load i32, ptr %arrayidx6, align 4
  %add = add nsw i32 %2, %1
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.false
  %cond = phi i32 [ %add, %cond.false ], [ %1, %entry ]
  store i32 %cond, ptr %arrayidx2, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @Test(ptr %res, ptr %c, ptr %d, ptr %p, i32 %n) #1 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 1600
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @Body(ptr %res, ptr undef, ptr %d, ptr %p, i32 %i.0), !llvm.access.group !0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !1

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK-LABEL: @Test
; CHECK: load i32,{{.*}}, !llvm.access.group !0
; CHECK: load i32,{{.*}}, !llvm.access.group !0
; CHECK: load i32,{{.*}}, !llvm.access.group !0
; CHECK: store i32{{.*}}, !llvm.access.group !0
; CHECK: br label %for.cond, !llvm.loop !1

attributes #0 = { norecurse nounwind uwtable }

!0 = distinct !{}
!1 = distinct !{!0, !{!"llvm.loop.parallel_accesses", !0}}
