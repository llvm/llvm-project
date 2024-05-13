; RUN: llc < %s -verify-coalescing
; PR12911
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

@d = common global i32 0, align 4
@c = common global i32 0, align 4
@b = common global i32 0, align 4
@h = common global i32 0, align 4
@f = common global i32 0, align 4
@g = common global i32 0, align 4
@a = common global i16 0, align 2
@e = common global i32 0, align 4

define void @fn1() nounwind uwtable ssp {
entry:
  %0 = load i32, ptr @d, align 4
  %tobool72 = icmp eq i32 %0, 0
  br i1 %tobool72, label %for.end32, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %1 = load i32, ptr @c, align 4
  %tobool2 = icmp eq i32 %1, 0
  %2 = load i32, ptr @b, align 4
  %cmp = icmp sgt i32 %2, 0
  %conv = zext i1 %cmp to i32
  %3 = load i32, ptr @g, align 4
  %tobool4 = icmp eq i32 %3, 0
  %4 = load i16, ptr @a, align 2
  %tobool9 = icmp eq i16 %4, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond25.loopexit.us-lcssa.us-lcssa, %if.end.us50, %if.end.us, %if.end.us.us, %for.cond1.preheader.lr.ph
  %j.073 = phi i32 [ undef, %for.cond1.preheader.lr.ph ], [ %j.1.us.us, %if.end.us.us ], [ %j.1.us, %if.end.us ], [ %j.073, %for.cond25.loopexit.us-lcssa.us-lcssa ], [ %j.1.us36, %if.end.us50 ]
  br i1 %tobool2, label %for.cond1.preheader.split.us, label %for.cond1.preheader.for.cond1.preheader.split_crit_edge

for.cond1.preheader.for.cond1.preheader.split_crit_edge: ; preds = %for.cond1.preheader
  br i1 %tobool9, label %if.end.us50, label %for.cond1.preheader.split.for.cond1.preheader.split.split_crit_edge

for.cond1.preheader.split.us:                     ; preds = %for.cond1.preheader
  br i1 %tobool9, label %cond.end.us.us, label %cond.end.us

cond.false18.us.us:                               ; preds = %if.end.us.us
  %5 = load i32, ptr @f, align 4
  %sext76 = shl i32 %5, 16
  %phitmp75 = ashr exact i32 %sext76, 16
  br label %cond.end.us.us

if.end.us.us:                                     ; preds = %cond.end.us.us, %if.then.us.us
  br i1 %tobool4, label %cond.false18.us.us, label %for.cond1.preheader

if.then.us.us:                                    ; preds = %cond.end.us.us
  store i32 0, ptr @f, align 4
  br label %if.end.us.us

cond.end.us.us:                                   ; preds = %cond.false18.us.us, %for.cond1.preheader.split.us
  %j.1.us.us = phi i32 [ %j.073, %for.cond1.preheader.split.us ], [ %phitmp75, %cond.false18.us.us ]
  store i32 %conv, ptr @h, align 4
  br i1 %cmp, label %if.then.us.us, label %if.end.us.us

cond.end21.us:                                    ; preds = %land.lhs.true12.us, %cond.false18.us
  %cond22.us = phi i16 [ %add.us, %cond.false18.us ], [ %4, %land.lhs.true12.us ]
  %conv24.us = sext i16 %cond22.us to i32
  br label %cond.end.us

cond.false18.us:                                  ; preds = %if.end6.us, %land.lhs.true12.us
  %add.us = add i16 %4, %conv7.us
  br label %cond.end21.us

land.lhs.true12.us:                               ; preds = %if.end6.us
  %conv10.us = sext i16 %conv7.us to i32
  %sub.us = sub nsw i32 0, %conv10.us
  %cmp14.us = icmp slt i32 %sub.us, 1
  br i1 %cmp14.us, label %cond.end21.us, label %cond.false18.us

if.end6.us:                                       ; preds = %if.end.us
  %6 = load i32, ptr @f, align 4
  %conv7.us = trunc i32 %6 to i16
  %tobool11.us = icmp eq i16 %conv7.us, 0
  br i1 %tobool11.us, label %cond.false18.us, label %land.lhs.true12.us

if.end.us:                                        ; preds = %cond.end.us, %if.then.us
  br i1 %tobool4, label %if.end6.us, label %for.cond1.preheader

if.then.us:                                       ; preds = %cond.end.us
  store i32 0, ptr @f, align 4
  br label %if.end.us

cond.end.us:                                      ; preds = %cond.end21.us, %for.cond1.preheader.split.us
  %j.1.us = phi i32 [ %conv24.us, %cond.end21.us ], [ %j.073, %for.cond1.preheader.split.us ]
  store i32 %conv, ptr @h, align 4
  br i1 %cmp, label %if.then.us, label %if.end.us

for.cond1.preheader.split.for.cond1.preheader.split.split_crit_edge: ; preds = %for.cond1.preheader.for.cond1.preheader.split_crit_edge
  br i1 %tobool4, label %if.end6.us65, label %for.cond25.loopexit.us-lcssa.us-lcssa

cond.false18.us40:                                ; preds = %if.end.us50
  %7 = load i32, ptr @f, align 4
  %sext = shl i32 %7, 16
  %phitmp = ashr exact i32 %sext, 16
  br label %if.end.us50

if.end.us50:                                      ; preds = %cond.false18.us40, %for.cond1.preheader.for.cond1.preheader.split_crit_edge
  %j.1.us36 = phi i32 [ %j.073, %for.cond1.preheader.for.cond1.preheader.split_crit_edge ], [ %phitmp, %cond.false18.us40 ]
  store i32 0, ptr @h, align 4
  br i1 %tobool4, label %cond.false18.us40, label %for.cond1.preheader

if.end6.us65:                                     ; preds = %if.end6.us65, %for.cond1.preheader.split.for.cond1.preheader.split.split_crit_edge
  store i32 0, ptr @h, align 4
  br label %if.end6.us65

for.cond25.loopexit.us-lcssa.us-lcssa:            ; preds = %for.cond1.preheader.split.for.cond1.preheader.split.split_crit_edge
  store i32 0, ptr @h, align 4
  br label %for.cond1.preheader

for.end32:                                        ; preds = %entry
  ret void
}
