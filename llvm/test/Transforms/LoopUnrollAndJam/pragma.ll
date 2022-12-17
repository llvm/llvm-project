; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-runtime < %s -S | FileCheck %s
; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-runtime -unroll-and-jam-threshold=15 < %s -S | FileCheck %s --check-prefix=CHECK-LOWTHRES

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK-LABEL: test1
; Basic check that these loops are by default UnJ'd
define void @test1(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK: %i.us = phi i32 [ %add8.us.{{[1-9]*}}, %for.latch ], [ 0, %for.outer.preheader.new ]
; CHECK-LOWTHRES: %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: nounroll_and_jam
; #pragma nounroll_and_jam
define void @nounroll_and_jam(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK: %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !1

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unroll_and_jam_count
; #pragma unroll_and_jam(8)
define void @unroll_and_jam_count(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK: %i.us = phi i32 [ %add8.us.7, %for.latch ], [ 0, %for.outer.preheader.new ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !3

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unroll_and_jam
; #pragma unroll_and_jam
define void @unroll_and_jam(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK: %i.us = phi i32 [ %add8.us.{{[1-9]*}}, %for.latch ], [ 0, %for.outer.preheader.new ]
; CHECK-LOWTHRES: %i.us = phi i32 [ %add8.us.{{[1-9]*}}, %for.latch ], [ 0, %for.outer.preheader.new ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !5

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: nounroll
; #pragma nounroll (which we take to mean disable unroll and jam too)
define void @nounroll(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK: %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !7

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unroll
; #pragma unroll (which we take to mean disable unroll and jam)
define void @unroll(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK: %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !9

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: nounroll_plus_unroll_and_jam
; #pragma clang loop nounroll, unroll_and_jam (which we take to mean do unroll_and_jam)
define void @nounroll_plus_unroll_and_jam(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK: %i.us = phi i32 [ %add8.us.{{[1-9]*}}, %for.latch ], [ 0, %for.outer.preheader.new ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !11

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


!1 = distinct !{!1, !2}
!2 = distinct !{!"llvm.loop.unroll_and_jam.disable"}
!3 = distinct !{!3, !4}
!4 = distinct !{!"llvm.loop.unroll_and_jam.count", i32 8}
!5 = distinct !{!5, !6}
!6 = distinct !{!"llvm.loop.unroll_and_jam.enable"}
!7 = distinct !{!7, !8}
!8 = distinct !{!"llvm.loop.unroll.disable"}
!9 = distinct !{!9, !10}
!10 = distinct !{!"llvm.loop.unroll.enable"}
!11 = distinct !{!11, !8, !6}
