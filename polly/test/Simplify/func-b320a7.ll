; RUN: opt %loadPolly -polly-print-simplify -polly-optree -disable-output < %s | FileCheck %s -match-full-lines

; llvm.org/PR47098
; Use-after-free by reference to Stmt remaining in InstStmtMap after removing it has been removed by Scop::simplifyScop.
; Removal happened in -polly-simplify, Reference was using in -polly-optree.
; Check that Simplify removes the definition of %0 as well of its use.

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.26.28806"

@"?var_27@@3JA" = external dso_local local_unnamed_addr global i32, align 4
@"?var_28@@3HA" = external dso_local local_unnamed_addr global i32, align 4

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @"?test@@YAXHEQEAY3M@1BI@BJ@H@Z"(i32 %a, i8 %b, ptr nocapture readonly %c) local_unnamed_addr {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %sext = shl i32 %a, 16
  %conv4 = ashr exact i32 %sext, 16
  %sub = add nsw i32 %conv4, -25941
  %cmp535 = icmp sgt i32 %sext, 1700069376
  br i1 %cmp535, label %for.cond8.preheader.lr.ph, label %for.cond.cleanup.critedge

for.cond8.preheader.lr.ph:                        ; preds = %entry.split
  %conv10 = zext i8 %b to i64
  %sub11 = add nsw i64 %conv10, -129
  %cmp1232.not = icmp eq i64 %sub11, 0
  br i1 %cmp1232.not, label %for.cond8.preheader, label %for.cond8.preheader.us

for.cond8.preheader.us:                           ; preds = %for.cond8.preheader.lr.ph, %for.cond8.for.cond.cleanup13_crit_edge.us
  %e.036.us = phi i16 [ %add.us, %for.cond8.for.cond.cleanup13_crit_edge.us ], [ 0, %for.cond8.preheader.lr.ph ]
  %idxprom.us = sext i16 %e.036.us to i64
  br label %for.body14.us

for.body14.us:                                    ; preds = %for.cond8.preheader.us, %for.body14.us
  %indvars.iv = phi i64 [ 0, %for.cond8.preheader.us ], [ %indvars.iv.next, %for.body14.us ]
  %arrayidx19.us = getelementptr inbounds [12 x [2 x [24 x [25 x i32]]]], ptr %c, i64 6, i64 2, i64 1, i64 %idxprom.us, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx19.us, align 4, !tbaa !3
  store i32 0, ptr @"?var_28@@3HA", align 4, !tbaa !3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %sub11
  br i1 %exitcond.not, label %for.cond8.for.cond.cleanup13_crit_edge.us, label %for.body14.us

for.cond8.for.cond.cleanup13_crit_edge.us:        ; preds = %for.body14.us
  %add.us = add i16 %e.036.us, 4
  %conv2.us = sext i16 %add.us to i32
  %cmp5.us = icmp sgt i32 %sub, %conv2.us
  br i1 %cmp5.us, label %for.cond8.preheader.us, label %for.cond.cleanup.critedge.loopexit38

for.cond.cleanup.critedge.loopexit38:             ; preds = %for.cond8.for.cond.cleanup13_crit_edge.us
  store i32 %0, ptr @"?var_27@@3JA", align 4, !tbaa !7
  br label %for.cond.cleanup.critedge

for.cond.cleanup.critedge:                        ; preds = %for.cond8.preheader, %for.cond.cleanup.critedge.loopexit38, %entry.split
  ret void

for.cond8.preheader:                              ; preds = %for.cond8.preheader.lr.ph, %for.cond8.preheader
  %e.036 = phi i16 [ %add, %for.cond8.preheader ], [ 0, %for.cond8.preheader.lr.ph ]
  %add = add i16 %e.036, 4
  %conv2 = sext i16 %add to i32
  %cmp5 = icmp sgt i32 %sub, %conv2
  br i1 %cmp5, label %for.cond8.preheader, label %for.cond.cleanup.critedge
}

!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"long", !5, i64 0}


; CHECK: Statistics {
; CHECK:     Empty domains removed: 2
; CHECK: }

; CHECK: After accesses {
; CHECK-NOT: Stmt_for_body14_us_a
; CHECK-NOT: Stmt_for_body14_us
; CHECK: }
