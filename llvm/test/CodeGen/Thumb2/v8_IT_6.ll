; RUN: llc < %s -mtriple=thumbv8 -show-mc-encoding | FileCheck %s
; CHECK-NOT: orrsne r0, r1 @ encoding: [0x08,0x43]
; Narrow tORR cannot be predicated and set CPSR at the same time!

declare void @f(i32)

define void @initCloneLookups() #1 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc24, %entry
  %cmp108 = phi i1 [ true, %entry ], [ %cmp, %for.inc24 ]
  %y.0105 = phi i32 [ 1, %entry ], [ %inc25, %for.inc24 ]
  %notlhs = icmp slt i32 %y.0105, 6
  %notlhs69 = icmp sgt i32 %y.0105, 4
  %sub = add nsw i32 %y.0105, -1
  %cmp1.i = icmp sgt i32 %sub, 5
  %cmp1.i54 = icmp sgt i32 %y.0105, 5
  br i1 %cmp108, label %if.then.us, label %for.cond1.preheader.for.cond1.preheader.split_crit_edge

for.cond1.preheader.for.cond1.preheader.split_crit_edge: ; preds = %for.cond1.preheader
  br i1 %notlhs, label %for.inc.us101, label %for.inc

if.then.us:                                       ; preds = %for.cond1.preheader, %for.inc.us
  %x.071.us = phi i32 [ %inc.us.pre-phi, %for.inc.us ], [ 1, %for.cond1.preheader ]
  %notrhs.us = icmp sge i32 %x.071.us, %y.0105
  %or.cond44.not.us = or i1 %notrhs.us, %notlhs
  %notrhs70.us = icmp sle i32 %x.071.us, %y.0105
  %tobool.us = or i1 %notrhs70.us, %notlhs69
  %or.cond66.us = and i1 %or.cond44.not.us, %tobool.us
  br i1 %or.cond66.us, label %getHexxagonIndex.exit52.us, label %if.then.us.for.inc.us_crit_edge

if.then.us.for.inc.us_crit_edge:                  ; preds = %if.then.us
  %inc.us.pre = add nsw i32 %x.071.us, 1
  br label %for.inc.us

getHexxagonIndex.exit52.us:                       ; preds = %if.then.us
  %cmp3.i.us = icmp slt i32 %x.071.us, 5
  %or.cond.i.us = and i1 %cmp1.i, %cmp3.i.us
  %..i.us = sext i1 %or.cond.i.us to i32
  tail call void @f(i32 %..i.us) #3
  %add.us = add nsw i32 %x.071.us, 1
  %cmp3.i55.us = icmp slt i32 %add.us, 5
  %or.cond.i56.us = and i1 %cmp1.i54, %cmp3.i55.us
  %..i57.us = sext i1 %or.cond.i56.us to i32
  tail call void @f(i32 %..i57.us) #3
  %or.cond.i48.us = and i1 %notlhs69, %cmp3.i55.us
  %..i49.us = sext i1 %or.cond.i48.us to i32
  tail call void @f(i32 %..i49.us) #3
  br label %for.inc.us

for.inc.us:                                       ; preds = %if.then.us.for.inc.us_crit_edge, %getHexxagonIndex.exit52.us
  %inc.us.pre-phi = phi i32 [ %inc.us.pre, %if.then.us.for.inc.us_crit_edge ], [ %add.us, %getHexxagonIndex.exit52.us ]
  %exitcond109 = icmp eq i32 %inc.us.pre-phi, 10
  br i1 %exitcond109, label %for.inc24, label %if.then.us

for.inc.us101:                                    ; preds = %for.cond1.preheader.for.cond1.preheader.split_crit_edge, %for.inc.us101
  %x.071.us74 = phi i32 [ %add.us89, %for.inc.us101 ], [ 1, %for.cond1.preheader.for.cond1.preheader.split_crit_edge ]
  %cmp3.i.us84 = icmp slt i32 %x.071.us74, 5
  %or.cond.i.us85 = and i1 %cmp1.i, %cmp3.i.us84
  %..i.us86 = sext i1 %or.cond.i.us85 to i32
  tail call void @f(i32 %..i.us86) #3
  %add.us89 = add nsw i32 %x.071.us74, 1
  %cmp3.i55.us93 = icmp slt i32 %add.us89, 5
  %or.cond.i56.us94 = and i1 %cmp1.i54, %cmp3.i55.us93
  %..i57.us95 = sext i1 %or.cond.i56.us94 to i32
  tail call void @f(i32 %..i57.us95) #3
  %or.cond.i48.us97 = and i1 %notlhs69, %cmp3.i55.us93
  %..i49.us98 = sext i1 %or.cond.i48.us97 to i32
  tail call void @f(i32 %..i49.us98) #3
  %exitcond110 = icmp eq i32 %add.us89, 10
  br i1 %exitcond110, label %for.inc24, label %for.inc.us101

for.inc:                                          ; preds = %for.cond1.preheader.for.cond1.preheader.split_crit_edge, %for.inc
  %x.071 = phi i32 [ %add, %for.inc ], [ 1, %for.cond1.preheader.for.cond1.preheader.split_crit_edge ]
  %cmp3.i = icmp slt i32 %x.071, 5
  %or.cond.i = and i1 %cmp1.i, %cmp3.i
  %..i = sext i1 %or.cond.i to i32
  tail call void @f(i32 %..i) #3
  %add = add nsw i32 %x.071, 1
  %cmp3.i55 = icmp slt i32 %add, 5
  %or.cond.i56 = and i1 %cmp1.i54, %cmp3.i55
  %..i57 = sext i1 %or.cond.i56 to i32
  tail call void @f(i32 %..i57) #3
  %or.cond.i48 = and i1 %notlhs69, %cmp3.i55
  %..i49 = sext i1 %or.cond.i48 to i32
  tail call void @f(i32 %..i49) #3
  %exitcond = icmp eq i32 %add, 10
  br i1 %exitcond, label %for.inc24, label %for.inc

for.inc24:                                        ; preds = %for.inc, %for.inc.us101, %for.inc.us
  %inc25 = add nsw i32 %y.0105, 1
  %cmp = icmp slt i32 %inc25, 10
  %exitcond111 = icmp eq i32 %inc25, 10
  br i1 %exitcond111, label %for.end26, label %for.cond1.preheader

for.end26:                                        ; preds = %for.inc24
  ret void
}

