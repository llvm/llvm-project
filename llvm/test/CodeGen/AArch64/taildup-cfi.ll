; REQUIRES: asserts
; RUN: llc -mtriple=arm64-unknown-linux-gnu -debug-only=tailduplication %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LINUX
; RUN: llc -mtriple=arm64-apple-darwin -debug-only=tailduplication %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DARWIN

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@g = common local_unnamed_addr global i32 0, align 4
@f = common local_unnamed_addr global i32 0, align 4
@a = common local_unnamed_addr global i32 0, align 4
@m = common local_unnamed_addr global i32 0, align 4
@l = common local_unnamed_addr global i32 0, align 4
@j = common local_unnamed_addr global i32 0, align 4
@k = common local_unnamed_addr global i32 0, align 4
@i = common local_unnamed_addr global i32 0, align 4
@d = common local_unnamed_addr global i32 0, align 4
@c = common local_unnamed_addr global i32 0, align 4
@e = common local_unnamed_addr global i32 0, align 4
@h = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind uwtable
define void @n(i32 %o, i32* nocapture readonly %b) local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* @g, align 4, !tbaa !2
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %entry.if.end_crit_edge, label %if.then

entry.if.end_crit_edge:                           ; preds = %entry
  %.pre = load i32, i32* @f, align 4, !tbaa !2
  br label %if.end

if.then:                                          ; preds = %entry
  store i32 0, i32* @f, align 4, !tbaa !2
  br label %if.end

; DARWIN-NOT:       Merging into block
; LINUX:    	      Merging into block

if.end:                                           ; preds = %entry.if.end_crit_edge, %if.then
  %1 = phi i32 [ %.pre, %entry.if.end_crit_edge ], [ 0, %if.then ]
  %cmp6 = icmp slt i32 %1, %o
  br i1 %cmp6, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %if.end
  %.pre7 = load i32, i32* @a, align 4, !tbaa !2
  %.pre8 = load i32, i32* @l, align 4, !tbaa !2
  %.pre9 = load i32, i32* @j, align 4, !tbaa !2
  %.pre10 = load i32, i32* @k, align 4, !tbaa !2
  %.pre11 = load i32, i32* @i, align 4, !tbaa !2
  br label %for.body

for.body:                                         ; preds = %if.end5, %for.body.lr.ph
  %2 = phi i32 [ %.pre11, %for.body.lr.ph ], [ %7, %if.end5 ]
  %3 = phi i32 [ %.pre10, %for.body.lr.ph ], [ %8, %if.end5 ]
  %4 = phi i32 [ %.pre9, %for.body.lr.ph ], [ %9, %if.end5 ]
  %5 = phi i32 [ %1, %for.body.lr.ph ], [ %inc, %if.end5 ]
  store i32 %.pre7, i32* @m, align 4, !tbaa !2
  %mul = mul nsw i32 %3, %4
  %cmp1 = icmp sgt i32 %.pre8, %mul
  %conv = zext i1 %cmp1 to i32
  %cmp2 = icmp slt i32 %2, %conv
  br i1 %cmp2, label %if.then4, label %if.end5

if.then4:                                         ; preds = %for.body
  %6 = load i32, i32* @d, align 4, !tbaa !2
  store i32 %6, i32* @k, align 4, !tbaa !2
  store i32 %6, i32* @i, align 4, !tbaa !2
  store i32 %6, i32* @j, align 4, !tbaa !2
  br label %if.end5

if.end5:                                          ; preds = %if.then4, %for.body
  %7 = phi i32 [ %6, %if.then4 ], [ %2, %for.body ]
  %8 = phi i32 [ %6, %if.then4 ], [ %3, %for.body ]
  %9 = phi i32 [ %6, %if.then4 ], [ %4, %for.body ]
  %10 = load i32, i32* @c, align 4, !tbaa !2
  %idxprom = sext i32 %10 to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  %11 = load i32, i32* %arrayidx, align 4, !tbaa !2
  %12 = load i32, i32* @e, align 4, !tbaa !2
  %sub = sub nsw i32 %11, %12
  store i32 %sub, i32* @h, align 4, !tbaa !2
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* @f, align 4, !tbaa !2
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %if.end5, %if.end
  ret void
}

attributes #0 = { norecurse nounwind uwtable }

!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{}
