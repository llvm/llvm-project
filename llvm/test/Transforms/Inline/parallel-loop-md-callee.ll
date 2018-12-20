; RUN: opt -S -inline < %s | FileCheck %s
;
; Check that the !llvm.access.group is still present after inlining.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @Body(i32* nocapture %res, i32* nocapture readnone %c, i32* nocapture readonly %d, i32* nocapture readonly %p, i32 %i) {
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4, !llvm.access.group !0
  %cmp = icmp eq i32 %0, 0
  %arrayidx2 = getelementptr inbounds i32, i32* %res, i64 %idxprom
  %1 = load i32, i32* %arrayidx2, align 4, !llvm.access.group !0
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:
  %arrayidx6 = getelementptr inbounds i32, i32* %d, i64 %idxprom
  %2 = load i32, i32* %arrayidx6, align 4, !llvm.access.group !0
  %add = add nsw i32 %2, %1
  br label %cond.end

cond.end:
  %cond = phi i32 [ %add, %cond.false ], [ %1, %entry ]
  store i32 %cond, i32* %arrayidx2, align 4
  ret void
}

define void @Test(i32* %res, i32* %c, i32* %d, i32* %p, i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 1600
  br i1 %cmp, label %for.body, label %for.end

for.body:
  call void @Body(i32* %res, i32* undef, i32* %d, i32* %p, i32 %i.0), !llvm.access.group !0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !1

for.end:
  ret void
}

!0 = distinct !{}                                          ; access group
!1 = distinct !{!1, !{!"llvm.loop.parallel_accesses", !0}} ; LoopID


; CHECK-LABEL: @Test
; CHECK: load i32,{{.*}}, !llvm.access.group !0
; CHECK: load i32,{{.*}}, !llvm.access.group !0
; CHECK: load i32,{{.*}}, !llvm.access.group !0
; CHECK: store i32 {{.*}}, !llvm.access.group !0
; CHECK: br label %for.cond, !llvm.loop !1
