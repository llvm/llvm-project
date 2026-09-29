; RUN: llc < %s -verify-coalescing
; PR14732
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10"

@c = common global i32 0, align 4
@b = common global i32 0, align 4
@a = common global i32 0, align 4
@d = common global i32 0, align 4

; This function creates an IMPLICIT_DEF with a long live range, even after
; ProcessImplicitDefs.
;
; The coalescer should be able to deal with all kinds of IMPLICIT_DEF live
; ranges, even if they are not common.

define void @f() nounwind uwtable ssp {
entry:
  %i = alloca i32, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc34, %entry
  %i.0.load44 = phi i32 [ %inc35, %for.inc34 ], [ undef, %entry ]
  %pi.0 = phi ptr [ %pi.4, %for.inc34 ], [ undef, %entry ]
  %tobool = icmp eq i32 %i.0.load44, 0
  br i1 %tobool, label %for.end36, label %for.body

for.body:                                         ; preds = %for.cond
  store i32 0, ptr @c, align 4
  br label %for.body2

for.body2:                                        ; preds = %for.body, %for.inc
  %i.0.load45 = phi i32 [ %i.0.load44, %for.body ], [ 0, %for.inc ]
  %tobool3 = icmp eq i32 %i.0.load45, 0
  br i1 %tobool3, label %if.then10, label %if.then

if.then:                                          ; preds = %for.body2
  store i32 0, ptr %i, align 4
  br label %for.body6

for.body6:                                        ; preds = %if.then, %for.body6
  store i32 0, ptr %i, align 4
  br i1 true, label %for.body6, label %for.inc

if.then10:                                        ; preds = %for.body2
  store i32 1, ptr @b, align 4
  ret void

for.inc:                                          ; preds = %for.body6
  br i1 undef, label %for.body2, label %if.end30

while.condthread-pre-split:                       ; preds = %label.loopexit, %while.condthread-pre-split.lr.ph.lr.ph, %for.inc27.backedge
  %0 = phi i32 [ %inc28, %for.inc27.backedge ], [ %inc285863, %while.condthread-pre-split.lr.ph.lr.ph ], [ %inc2858, %label.loopexit ]
  %inc2060 = phi i32 [ %inc20, %for.inc27.backedge ], [ %a.promoted.pre, %while.condthread-pre-split.lr.ph.lr.ph ], [ %inc20, %label.loopexit ]
  br label %while.cond

while.cond:                                       ; preds = %while.condthread-pre-split, %while.cond
  %p2.1.in = phi ptr [ %pi.3.ph, %while.cond ], [ %i, %while.condthread-pre-split ]
  br i1 %tobool19, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  %inc20 = add nsw i32 %inc2060, 1
  %tobool21 = icmp eq i32 %inc2060, 0
  br i1 %tobool21, label %for.inc27.backedge, label %if.then22

for.inc27.backedge:                               ; preds = %while.end, %if.then22
  %inc28 = add nsw i32 %0, 1
  store i32 %inc28, ptr @b, align 4
  %tobool17 = icmp eq i32 %inc28, 0
  br i1 %tobool17, label %for.inc27.if.end30.loopexit56_crit_edge, label %while.condthread-pre-split

if.then22:                                        ; preds = %while.end
  %1 = load i16, ptr %p2.1.in, align 2
  %tobool23 = icmp eq i16 %1, 0
  br i1 %tobool23, label %for.inc27.backedge, label %label.loopexit

label.loopexit:                                   ; preds = %if.then22
  store i32 %inc20, ptr @a, align 4
  %inc2858 = add nsw i32 %0, 1
  store i32 %inc2858, ptr @b, align 4
  %tobool1759 = icmp eq i32 %inc2858, 0
  br i1 %tobool1759, label %if.end30, label %while.condthread-pre-split

for.inc27.if.end30.loopexit56_crit_edge:          ; preds = %for.inc27.backedge
  store i32 %inc20, ptr @a, align 4
  br label %if.end30

if.end30:                                         ; preds = %for.inc27.if.end30.loopexit56_crit_edge, %label.loopexit, %label.preheader, %for.inc
  %i.0.load46 = phi i32 [ 0, %for.inc ], [ %i.0.load4669, %label.preheader ], [ %i.0.load4669, %label.loopexit ], [ %i.0.load4669, %for.inc27.if.end30.loopexit56_crit_edge ]
  %pi.4 = phi ptr [ %i, %for.inc ], [ %pi.3.ph, %label.preheader ], [ %pi.3.ph, %label.loopexit ], [ %pi.3.ph, %for.inc27.if.end30.loopexit56_crit_edge ]
  %2 = load i32, ptr %pi.4, align 4
  %tobool31 = icmp eq i32 %2, 0
  br i1 %tobool31, label %for.inc34, label %label.preheader

for.inc34:                                        ; preds = %if.end30
  %inc35 = add nsw i32 %i.0.load46, 1
  store i32 %inc35, ptr %i, align 4
  br label %for.cond

for.end36:                                        ; preds = %for.cond
  store i32 1, ptr %i, align 4
  %3 = load i32, ptr @c, align 4
  %tobool37 = icmp eq i32 %3, 0
  br i1 %tobool37, label %label.preheader, label %land.rhs

land.rhs:                                         ; preds = %for.end36
  store i32 0, ptr @a, align 4
  br label %label.preheader

label.preheader:                                  ; preds = %for.end36, %if.end30, %land.rhs
  %i.0.load4669 = phi i32 [ 1, %land.rhs ], [ %i.0.load46, %if.end30 ], [ 1, %for.end36 ]
  %pi.3.ph = phi ptr [ %pi.0, %land.rhs ], [ %pi.4, %if.end30 ], [ %pi.0, %for.end36 ]
  %4 = load i32, ptr @b, align 4
  %inc285863 = add nsw i32 %4, 1
  store i32 %inc285863, ptr @b, align 4
  %tobool175964 = icmp eq i32 %inc285863, 0
  br i1 %tobool175964, label %if.end30, label %while.condthread-pre-split.lr.ph.lr.ph

while.condthread-pre-split.lr.ph.lr.ph:           ; preds = %label.preheader
  %.pr50 = load i32, ptr @d, align 4
  %tobool19 = icmp eq i32 %.pr50, 0
  %a.promoted.pre = load i32, ptr @a, align 4
  br label %while.condthread-pre-split
}
