; RUN: opt -loop-unroll-and-jam -allow-unroll-and-jam -unroll-runtime -unroll-partial-threshold=60 < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=tbaa,basic-aa -passes='loop-unroll-and-jam' -allow-unroll-and-jam -unroll-runtime -unroll-partial-threshold=60 < %s -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: function
; The explicit metadata here should force this to be unroll and jammed 4 times (hence the %.pre60.3)
; CHECK: %.pre = phi i8 [ %.pre60.3, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %.pre.pre, %for.cond1.preheader.us.preheader.new ]
; CHECK: %indvars.iv.3 = phi i64 [ 0, %for.cond1.preheader.us ], [ %indvars.iv.next.3, %for.body4.us ]
define void @function(i8* noalias nocapture %dst, i32 %dst_stride, i8* noalias nocapture readonly %src, i32 %src_stride, i32 %A, i32 %B, i32 %C, i32 %D, i32 %width, i32 %height) {
entry:
  %idxprom = sext i32 %src_stride to i64
  %cmp52 = icmp sgt i32 %height, 0
  br i1 %cmp52, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp249 = icmp sgt i32 %width, 0
  %idx.ext = sext i32 %dst_stride to i64
  br i1 %cmp249, label %for.cond1.preheader.us.preheader, label %for.cond.cleanup

for.cond1.preheader.us.preheader:                 ; preds = %for.cond1.preheader.lr.ph
  %.pre.pre = load i8, i8* %src, align 1
  %wide.trip.count = zext i32 %width to i64
  br label %for.cond1.preheader.us

for.cond1.preheader.us:                           ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %for.cond1.preheader.us.preheader
  %.pre = phi i8 [ %.pre60, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %.pre.pre, %for.cond1.preheader.us.preheader ]
  %srcp.056.us.pn = phi i8* [ %srcp.056.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %src, %for.cond1.preheader.us.preheader ]
  %y.055.us = phi i32 [ %inc30.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %dst.addr.054.us = phi i8* [ %add.ptr.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %dst, %for.cond1.preheader.us.preheader ]
  %srcp.056.us = getelementptr inbounds i8, i8* %srcp.056.us.pn, i64 %idxprom
  %.pre60 = load i8, i8* %srcp.056.us, align 1
  br label %for.body4.us

for.body4.us:                                     ; preds = %for.body4.us, %for.cond1.preheader.us
  %0 = phi i8 [ %.pre60, %for.cond1.preheader.us ], [ %3, %for.body4.us ]
  %1 = phi i8 [ %.pre, %for.cond1.preheader.us ], [ %2, %for.body4.us ]
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader.us ], [ %indvars.iv.next, %for.body4.us ]
  %conv.us = zext i8 %1 to i32
  %mul.us = mul nsw i32 %conv.us, %A
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx8.us = getelementptr inbounds i8, i8* %srcp.056.us.pn, i64 %indvars.iv.next
  %2 = load i8, i8* %arrayidx8.us, align 1
  %conv9.us = zext i8 %2 to i32
  %mul10.us = mul nsw i32 %conv9.us, %B
  %conv14.us = zext i8 %0 to i32
  %mul15.us = mul nsw i32 %conv14.us, %C
  %arrayidx19.us = getelementptr inbounds i8, i8* %srcp.056.us, i64 %indvars.iv.next
  %3 = load i8, i8* %arrayidx19.us, align 1
  %conv20.us = zext i8 %3 to i32
  %mul21.us = mul nsw i32 %conv20.us, %D
  %add11.us = add i32 %mul.us, 32
  %add16.us = add i32 %add11.us, %mul10.us
  %add22.us = add i32 %add16.us, %mul15.us
  %add23.us = add i32 %add22.us, %mul21.us
  %4 = lshr i32 %add23.us, 6
  %conv24.us = trunc i32 %4 to i8
  %arrayidx26.us = getelementptr inbounds i8, i8* %dst.addr.054.us, i64 %indvars.iv
  store i8 %conv24.us, i8* %arrayidx26.us, align 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us

for.cond1.for.cond.cleanup3_crit_edge.us:         ; preds = %for.body4.us
  %add.ptr.us = getelementptr inbounds i8, i8* %dst.addr.054.us, i64 %idx.ext
  %inc30.us = add nuw nsw i32 %y.055.us, 1
  %exitcond58 = icmp eq i32 %inc30.us, %height
  br i1 %exitcond58, label %for.cond.cleanup, label %for.cond1.preheader.us, !llvm.loop !5

for.cond.cleanup:                                 ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %for.cond1.preheader.lr.ph, %entry
  ret void
}

; CHECK-LABEL: function2
; The explicit metadata here should force this to be unroll and jammed, but
; the count is left to thresholds. In this case 2 (hence %.pre60.1).
; CHECK: %.pre = phi i8 [ %.pre60.1, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %.pre.pre, %for.cond1.preheader.us.preheader.new ]
; CHECK: %indvars.iv.1 = phi i64 [ 0, %for.cond1.preheader.us ], [ %indvars.iv.next.1, %for.body4.us ]
define void @function2(i8* noalias nocapture %dst, i32 %dst_stride, i8* noalias nocapture readonly %src, i32 %src_stride, i32 %A, i32 %B, i32 %C, i32 %D, i32 %width, i32 %height) {
entry:
  %idxprom = sext i32 %src_stride to i64
  %cmp52 = icmp sgt i32 %height, 0
  br i1 %cmp52, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp249 = icmp sgt i32 %width, 0
  %idx.ext = sext i32 %dst_stride to i64
  br i1 %cmp249, label %for.cond1.preheader.us.preheader, label %for.cond.cleanup

for.cond1.preheader.us.preheader:                 ; preds = %for.cond1.preheader.lr.ph
  %.pre.pre = load i8, i8* %src, align 1
  %wide.trip.count = zext i32 %width to i64
  br label %for.cond1.preheader.us

for.cond1.preheader.us:                           ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %for.cond1.preheader.us.preheader
  %.pre = phi i8 [ %.pre60, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %.pre.pre, %for.cond1.preheader.us.preheader ]
  %srcp.056.us.pn = phi i8* [ %srcp.056.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %src, %for.cond1.preheader.us.preheader ]
  %y.055.us = phi i32 [ %inc30.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %dst.addr.054.us = phi i8* [ %add.ptr.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %dst, %for.cond1.preheader.us.preheader ]
  %srcp.056.us = getelementptr inbounds i8, i8* %srcp.056.us.pn, i64 %idxprom
  %.pre60 = load i8, i8* %srcp.056.us, align 1
  br label %for.body4.us

for.body4.us:                                     ; preds = %for.body4.us, %for.cond1.preheader.us
  %0 = phi i8 [ %.pre60, %for.cond1.preheader.us ], [ %3, %for.body4.us ]
  %1 = phi i8 [ %.pre, %for.cond1.preheader.us ], [ %2, %for.body4.us ]
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader.us ], [ %indvars.iv.next, %for.body4.us ]
  %conv.us = zext i8 %1 to i32
  %mul.us = mul nsw i32 %conv.us, %A
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx8.us = getelementptr inbounds i8, i8* %srcp.056.us.pn, i64 %indvars.iv.next
  %2 = load i8, i8* %arrayidx8.us, align 1
  %conv9.us = zext i8 %2 to i32
  %mul10.us = mul nsw i32 %conv9.us, %B
  %conv14.us = zext i8 %0 to i32
  %mul15.us = mul nsw i32 %conv14.us, %C
  %arrayidx19.us = getelementptr inbounds i8, i8* %srcp.056.us, i64 %indvars.iv.next
  %3 = load i8, i8* %arrayidx19.us, align 1
  %conv20.us = zext i8 %3 to i32
  %mul21.us = mul nsw i32 %conv20.us, %D
  %add11.us = add i32 %mul.us, 32
  %add16.us = add i32 %add11.us, %mul10.us
  %add22.us = add i32 %add16.us, %mul15.us
  %add23.us = add i32 %add22.us, %mul21.us
  %4 = lshr i32 %add23.us, 6
  %conv24.us = trunc i32 %4 to i8
  %arrayidx26.us = getelementptr inbounds i8, i8* %dst.addr.054.us, i64 %indvars.iv
  store i8 %conv24.us, i8* %arrayidx26.us, align 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us

for.cond1.for.cond.cleanup3_crit_edge.us:         ; preds = %for.body4.us
  %add.ptr.us = getelementptr inbounds i8, i8* %dst.addr.054.us, i64 %idx.ext
  %inc30.us = add nuw nsw i32 %y.055.us, 1
  %exitcond58 = icmp eq i32 %inc30.us, %height
  br i1 %exitcond58, label %for.cond.cleanup, label %for.cond1.preheader.us, !llvm.loop !7

for.cond.cleanup:                                 ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %for.cond1.preheader.lr.ph, %entry
  ret void
}

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll_and_jam.count", i32 4}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll_and_jam.enable"}
