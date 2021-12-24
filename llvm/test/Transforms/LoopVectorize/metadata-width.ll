; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-target-supports-scalable-vectors=true -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; CHECK-LABEL: @test1(
; CHECK: store <8 x i32>
; CHECK: ret void
define void @test1(i32* nocapture %a, i32 %n) #0 {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 42, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: @test2(
; CHECK: store <vscale x 8 x i32>
; CHECK: ret void
define void @test2(i32* nocapture %a, i32 %n) #0 {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 42, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !2

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: @test3(
; CHECK: store <8 x i32>
; CHECK: ret void
define void @test3(i32* nocapture %a, i32 %n) #0 {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 42, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Test that when only the width property is specified, and not the scalable.enable flag,
; that the hint defaults to 'scalable.enable = false'.

; CHECK-LABEL: @test4(
; CHECK: store <8 x i32>
; CHECK: ret void
define void @test4(i32* nocapture %a, i32 %n) #0 {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 42, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !6

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!0, !1, !5}
!1 = !{!"llvm.loop.vectorize.width", i32 8}
!2 = !{!2, !1, !3}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i32 1}
!4 = !{!4, !1, !5}
!5 = !{!"llvm.loop.vectorize.scalable.enable", i32 0}
!6 = !{!6, !7}
!7 = !{!"llvm.loop.vectorize.width", i32 8}
