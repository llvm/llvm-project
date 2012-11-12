; RUN: llc < %s -o /dev/null -stats 2>&1 | FileCheck %s -check-prefix=STATS
; Radar 10266272
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios4.0.0"
; STATS-NOT: machine-sink

define i32 @foo(i32 %h) nounwind readonly ssp {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %cmp = icmp slt i32 0, %h
  br i1 %cmp, label %for.body, label %if.end299

for.body:                                         ; preds = %for.cond
  %v.5 = select i1 undef, i32 undef, i32 0
  %0 = load i8* undef, align 1, !tbaa !0
  %conv88 = zext i8 %0 to i32
  %sub89 = sub nsw i32 0, %conv88
  %v.8 = select i1 undef, i32 undef, i32 %sub89
  %1 = load i8* null, align 1, !tbaa !0
  %conv108 = zext i8 %1 to i32
  %2 = load i8* undef, align 1, !tbaa !0
  %conv110 = zext i8 %2 to i32
  %sub111 = sub nsw i32 %conv108, %conv110
  %cmp112 = icmp slt i32 %sub111, 0
  %sub115 = sub nsw i32 0, %sub111
  %v.10 = select i1 %cmp112, i32 %sub115, i32 %sub111
  %add62 = add i32 0, %v.5
  %add73 = add i32 %add62, 0
  %add84 = add i32 %add73, 0
  %add95 = add i32 %add84, %v.8
  %add106 = add i32 %add95, 0
  %add117 = add i32 %add106, %v.10
  %add128 = add i32 %add117, 0
  %add139 = add i32 %add128, 0
  %add150 = add i32 %add139, 0
  %add161 = add i32 %add150, 0
  %add172 = add i32 %add161, 0
  br i1 undef, label %for.cond, label %if.end299

if.end299:                                        ; preds = %for.body, %for.cond
  %s.10 = phi i32 [ %add172, %for.body ], [ 0, %for.cond ]
  ret i32 %s.10
}

!0 = metadata !{metadata !"omnipotent char", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA", null}
