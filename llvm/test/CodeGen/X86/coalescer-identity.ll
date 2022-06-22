; RUN: llc < %s -verify-coalescing
; PR12927
target triple = "x86_64-apple-macosx10.8.0"

; This is a case where removeCopyByCommutingDef() creates an identity copy that
; joinCopy must then deal with correctly.

@s = common global i16 0, align 2
@g1 = common global i32 0, align 4
@g2 = common global i32 0, align 4
@g0 = common global i32 0, align 4

define void @func() nounwind uwtable ssp {
for.body.lr.ph:
  %0 = load i32, ptr @g2, align 4
  %tobool6 = icmp eq i32 %0, 0
  %s.promoted = load i16, ptr @s, align 2
  %.pre = load i32, ptr @g1, align 4
  br i1 %tobool6, label %for.body.us, label %for.body

for.body.us:                                      ; preds = %for.body.lr.ph, %for.inc.us
  %1 = phi i32 [ %3, %for.inc.us ], [ %.pre, %for.body.lr.ph ]
  %dec13.us = phi i16 [ %dec12.us, %for.inc.us ], [ %s.promoted, %for.body.lr.ph ]
  %i.011.us = phi i32 [ %inc.us, %for.inc.us ], [ undef, %for.body.lr.ph ]
  %v.010.us = phi i32 [ %phitmp.us, %for.inc.us ], [ 1, %for.body.lr.ph ]
  %tobool1.us = icmp ne i32 %v.010.us, 0
  %2 = zext i1 %tobool1.us to i16
  %lnot.ext.us = xor i16 %2, 1
  %add.us = add i16 %dec13.us, %lnot.ext.us
  %conv3.us = zext i16 %add.us to i32
  %add4.us = sub i32 0, %1
  %tobool5.us = icmp eq i32 %conv3.us, %add4.us
  br i1 %tobool5.us, label %for.inc.us, label %if.then7.us

for.inc.us:                                       ; preds = %cond.end.us, %for.body.us
  %3 = phi i32 [ %1, %for.body.us ], [ %4, %cond.end.us ]
  %dec12.us = phi i16 [ %add.us, %for.body.us ], [ %dec.us, %cond.end.us ]
  %inc.us = add i32 %i.011.us, 1
  %phitmp.us = udiv i32 %v.010.us, 12
  %tobool.us = icmp eq i32 %inc.us, 0
  br i1 %tobool.us, label %for.end, label %for.body.us

cond.end.us:                                      ; preds = %if.then7.us, %cond.false.us
  %4 = phi i32 [ 0, %cond.false.us ], [ %1, %if.then7.us ]
  %cond.us = phi i32 [ 0, %cond.false.us ], [ %v.010.us, %if.then7.us ]
  store i32 %cond.us, ptr @g0, align 4
  br label %for.inc.us

cond.false.us:                                    ; preds = %if.then7.us
  store i32 0, ptr @g1, align 4
  br label %cond.end.us

if.then7.us:                                      ; preds = %for.body.us
  %dec.us = add i16 %add.us, -1
  br i1 %tobool1.us, label %cond.end.us, label %cond.false.us

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %dec13 = phi i16 [ %dec12, %for.body ], [ %s.promoted, %for.body.lr.ph ]
  %i.011 = phi i32 [ %inc, %for.body ], [ undef, %for.body.lr.ph ]
  %v.010 = phi i32 [ %phitmp, %for.body ], [ 1, %for.body.lr.ph ]
  %tobool1 = icmp eq i32 %v.010, 0
  %lnot.ext = zext i1 %tobool1 to i16
  %add = add i16 %dec13, %lnot.ext
  %conv3 = zext i16 %add to i32
  %add4 = sub i32 0, %.pre
  %not.tobool5 = icmp ne i32 %conv3, %add4
  %dec = sext i1 %not.tobool5 to i16
  %dec12 = add i16 %add, %dec
  %inc = add i32 %i.011, 1
  %phitmp = udiv i32 %v.010, 12
  %tobool = icmp eq i32 %inc, 0
  br i1 %tobool, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc.us, %for.body
  %dec12.lcssa = phi i16 [ %dec12.us, %for.inc.us ], [ %dec12, %for.body ]
  store i16 %dec12.lcssa, ptr @s, align 2
  ret void
}
