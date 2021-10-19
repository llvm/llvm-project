; RUN: opt < %s -S -passes="default<O2>" -unroll-runtime=true -enable-unroll-and-jam -unroll-threshold-default=0 -unroll-threshold-aggressive=300 | FileCheck %s -check-prefix=O2
; RUN: opt < %s -S -passes="default<O3>" -unroll-runtime=true -enable-unroll-and-jam -unroll-threshold-default=0 -unroll-threshold-aggressive=300 | FileCheck %s -check-prefix=O3
; RUN: opt < %s -S -passes="default<Os>" -unroll-runtime=true -enable-unroll-and-jam -unroll-threshold-default=0 -unroll-threshold-aggressive=300 | FileCheck %s -check-prefix=Os
; RUN: opt < %s -S -passes="default<Oz>" -unroll-runtime=true -enable-unroll-and-jam -unroll-threshold-default=0 -unroll-threshold-aggressive=300 | FileCheck %s -check-prefix=Oz

; Check that Os and Oz are optimized like O2, not like O3. To easily highlight
; the behavior, we artificially disable unrolling for anything but O3 by setting
; the default threshold to 0.

; O3:     for.inner.1
; O2-NOT: for.inner.1
; Os-NOT: for.inner.1
; Oz-NOT: for.inner.1

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @test1(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
entry:
  %cmp = icmp ne i32 %J, 0
  %cmpJ = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmpJ
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !5
  %add = add i32 %0, %sum
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4, !tbaa !5
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}



!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
