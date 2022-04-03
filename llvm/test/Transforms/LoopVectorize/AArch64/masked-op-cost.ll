; REQUIRES: asserts
; RUN: opt -loop-vectorize -force-vector-interleave=1 -S -debug-only=loop-vectorize < %s 2>%t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix=CHECK-COST

target triple = "aarch64-unknown-linux-gnu"

; CHECK-COST: Checking a loop in 'fixed_width'
; CHECK-COST: Found an estimated cost of 11 for VF 2 For instruction:   store i32 2, i32* %arrayidx1, align 4
; CHECK-COST: Found an estimated cost of 25 for VF 4 For instruction:   store i32 2, i32* %arrayidx1, align 4
; CHECK-COST: Selecting VF: 1.

; We should decide this loop is not worth vectorising using fixed width vectors
define void @fixed_width(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i64 %n) #0 {
; CHECK-LABEL: @fixed_width(
; CHECK-NOT: vector.body
entry:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.inc
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %i.07 = phi i64 [ %inc, %for.inc ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %i.07
  %0 = load i32, i32* %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 %i.07
  store i32 2, i32* %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !5
}


; CHECK-COST: Checking a loop in 'scalable'
; CHECK-COST: Found an estimated cost of 2 for VF vscale x 4 For instruction:   store i32 2, i32* %arrayidx1, align 4

define void @scalable(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i64 %n) #0 {
; CHECK-LABEL: @scalable(
; CHECK: vector.body
; CHECK: call void @llvm.masked.store.nxv4i32.p0nxv4i32
entry:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.inc
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %i.07 = phi i64 [ %inc, %for.inc ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %i.07
  %0 = load i32, i32* %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 %i.07
  store i32 2, i32* %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !0
}

attributes #0 = { "target-features"="+neon,+sve" }

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
