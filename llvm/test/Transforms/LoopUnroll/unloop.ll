; RUN: opt < %s -S -loop-unroll -verify-loop-info | FileCheck %s
;
; Unit tests for LoopInfo::updateUnloop.

declare i1 @check() nounwind

; Ensure that tail->inner is removed and rely on verify-loopinfo to
; check soundness.
;
; CHECK: @skiplevelexit
; CHECK: tail:
; CHECK-NOT: br
; CHECK: ret void
define void @skiplevelexit() nounwind {
entry:
  br label %outer

outer:
  br label %inner

inner:
  %iv = phi i32 [ 0, %outer ], [ %inc, %tail ]
  %inc = add i32 %iv, 1
  call zeroext i1 @check()
  br i1 true, label %outer.backedge, label %tail

tail:
  br i1 false, label %inner, label %exit

outer.backedge:
  br label %outer

exit:
  ret void
}

; Remove the middle loop of a triply nested loop tree.
; Ensure that only the middle loop is removed and rely on verify-loopinfo to
; check soundness.
;
; CHECK: @unloopNested
; Outer loop control.
; CHECK: while.body:
; CHECK: br i1 %cmp3, label %if.then, label %if.end
; Inner loop control.
; CHECK: while.end14.i:
; CHECK: br i1 %call15.i, label %if.end.i, label %exit
; Middle loop control should no longer reach %while.cond.
; Now it is the outer loop backedge.
; CHECK: exit:
; CHECK: br label %while.cond.outer
define void @unloopNested() {
entry:
  br label %while.cond.outer

while.cond.outer:
  br label %while.cond

while.cond:
  %cmp = call zeroext i1 @check()
  br i1 %cmp, label %while.body, label %while.end

while.body:
  %cmp3 = call zeroext i1 @check()
  br i1 %cmp3, label %if.then, label %if.end

if.then:
  br label %return

if.end:
  %cmp.i48 = call zeroext i1 @check()
  br i1 %cmp.i48, label %if.then.i, label %if.else20.i

if.then.i:
  %cmp8.i = call zeroext i1 @check()
  br i1 %cmp8.i, label %merge, label %if.else.i

if.else.i:
  br label %merge

if.else20.i:
  %cmp25.i = call zeroext i1 @check()
  br i1 %cmp25.i, label %merge, label %if.else28.i

if.else28.i:
  br label %merge

merge:
  br label %while.cond2.i

while.cond2.i:
  %cmp.i = call zeroext i1 @check()
  br i1 %cmp.i, label %while.cond2.backedge.i, label %while.end.i

while.cond2.backedge.i:
  br label %while.cond2.i

while.end.i:
  %cmp1114.i = call zeroext i1 @check()
  br i1 %cmp1114.i, label %while.body12.lr.ph.i, label %while.end14.i

while.body12.lr.ph.i:
  br label %while.end14.i

while.end14.i:
  %call15.i = call zeroext i1 @check()
  br i1 %call15.i, label %if.end.i, label %exit

if.end.i:
  br label %while.cond2.backedge.i

exit:
  br i1 false, label %while.cond, label %if.else

if.else:
  br label %while.cond.outer

while.end:
  br label %return

return:
  ret void
}

; Remove the middle loop of a deeply nested loop tree.
; Ensure that only the middle loop is removed and rely on verify-loopinfo to
; check soundness.
;
; This test must be disabled until trip count computation can be optimized...
; rdar:14038809 [SCEV]: Optimize trip count computation for multi-exit loops.
; CHECKFIXME: @unloopDeepNested
; Inner-inner loop control.
; CHECKFIXME: while.cond.us.i:
; CHECKFIXME: br i1 %cmp.us.i, label %next_data.exit, label %while.body.us.i
; CHECKFIXME: if.then.us.i:
; CHECKFIXME: br label %while.cond.us.i
; Inner loop tail.
; CHECKFIXME: if.else.i:
; CHECKFIXME: br label %while.cond.outer.i
; Middle loop control (removed).
; CHECKFIXME: valid_data.exit:
; CHECKFIXME-NOT: br
; CHECKFIXME: %cmp = call zeroext i1 @check()
; Outer loop control.
; CHECKFIXME: copy_data.exit:
; CHECKFIXME: br i1 %cmp38, label %if.then39, label %while.cond.outer
; Outer-outer loop tail.
; CHECKFIXME: while.cond.outer.outer.backedge:
; CHECKFIXME: br label %while.cond.outer.outer
define void @unloopDeepNested() nounwind {
for.cond8.preheader.i:
  %cmp113.i = call zeroext i1 @check()
  br i1 %cmp113.i, label %make_data.exit, label %for.body13.lr.ph.i

for.body13.lr.ph.i:
  br label %make_data.exit

make_data.exit:
  br label %while.cond.outer.outer

while.cond.outer.outer:
  br label %while.cond.outer

while.cond.outer:
  br label %while.cond

while.cond:
  br label %while.cond.outer.i

while.cond.outer.i:
  %tmp192.ph.i = call zeroext i1 @check()
  br i1 %tmp192.ph.i, label %while.cond.outer.split.us.i, label %while.body.loopexit

while.cond.outer.split.us.i:
  br label %while.cond.us.i

while.cond.us.i:
  %cmp.us.i = call zeroext i1 @check()
  br i1 %cmp.us.i, label %next_data.exit, label %while.body.us.i

while.body.us.i:
  %cmp7.us.i = call zeroext i1 @check()
  br i1 %cmp7.us.i, label %if.then.us.i, label %if.else.i

if.then.us.i:
  br label %while.cond.us.i

if.else.i:
  br label %while.cond.outer.i

next_data.exit:
  %tmp192.ph.i.lcssa28 = call zeroext i1 @check()
  br i1 %tmp192.ph.i.lcssa28, label %while.end, label %while.body

while.body.loopexit:
  br label %while.body

while.body:
  br label %while.cond.i

while.cond.i:
  %cmp.i = call zeroext i1 @check()
  br i1 %cmp.i, label %valid_data.exit, label %while.body.i

while.body.i:
  %cmp7.i = call zeroext i1 @check()
  br i1 %cmp7.i, label %valid_data.exit, label %if.end.i

if.end.i:
  br label %while.cond.i

valid_data.exit:
  br i1 true, label %if.then, label %while.cond

if.then:
  %cmp = call zeroext i1 @check()
  br i1 %cmp, label %if.then12, label %if.end

if.then12:
  br label %if.end

if.end:
  %tobool3.i = call zeroext i1 @check()
  br i1 %tobool3.i, label %copy_data.exit, label %while.body.lr.ph.i

while.body.lr.ph.i:
  br label %copy_data.exit

copy_data.exit:
  %cmp38 = call zeroext i1 @check()
  br i1 %cmp38, label %if.then39, label %while.cond.outer

if.then39:
  %cmp5.i = call zeroext i1 @check()
  br i1 %cmp5.i, label %while.cond.outer.outer.backedge, label %for.cond8.preheader.i8.thread

for.cond8.preheader.i8.thread:
  br label %while.cond.outer.outer.backedge

while.cond.outer.outer.backedge:
  br label %while.cond.outer.outer

while.end:
  ret void
}

; Remove a nested loop with irreducible control flow.
; Ensure that only the middle loop is removed and rely on verify-loopinfo to
; check soundness.
;
; CHECK: @unloopIrreducible
; Irreducible loop.
; CHECK: for.inc117:
; CHECK: br label %for.cond103t
; Nested loop (removed).
; CHECK: for.inc159:
; CHECK: br label %for.inc163
define void @unloopIrreducible() nounwind {

entry:
  br label %for.body

for.body:
  %cmp2113 = call zeroext i1 @check()
  br i1 %cmp2113, label %for.body22.lr.ph, label %for.inc163

for.body22.lr.ph:
  br label %for.body22

for.body22:
  br label %for.body33

for.body33:
  br label %for.end

for.end:
  %cmp424 = call zeroext i1 @check()
  br i1 %cmp424, label %for.body43.lr.ph, label %for.end93

for.body43.lr.ph:
  br label %for.end93

for.end93:
  %cmp96 = call zeroext i1 @check()
  br i1 %cmp96, label %if.then97, label %for.cond103

if.then97:
  br label %for.cond103t

for.cond103t:
  br label %for.cond103

for.cond103:
  %cmp105 = call zeroext i1 @check()
  br i1 %cmp105, label %for.body106, label %for.end120

for.body106:
  %cmp108 = call zeroext i1 @check()
  br i1 %cmp108, label %if.then109, label %for.inc117

if.then109:
  br label %for.inc117

for.inc117:
  br label %for.cond103t

for.end120:
  br label %for.inc159

for.inc159:
  br i1 false, label %for.body22, label %for.cond15.for.inc163_crit_edge

for.cond15.for.inc163_crit_edge:
  br label %for.inc163

for.inc163:
  %cmp12 = call zeroext i1 @check()
  br i1 %cmp12, label %for.body, label %for.end166

for.end166:
  ret void

}

; Remove a loop whose exit branches into a sibling loop.
; Ensure that only the loop is removed and rely on verify-loopinfo to
; check soundness.
;
; CHECK: @unloopCriticalEdge
; CHECK: while.cond.outer.i.loopexit.split:
; CHECK: br label %while.body
; CHECK: while.body:
; CHECK: br label %for.end78
define void @unloopCriticalEdge() nounwind {
entry:
  br label %for.cond31

for.cond31:
  br i1 undef, label %for.body35, label %for.end94

for.body35:
  br label %while.cond.i.preheader

while.cond.i.preheader:
  br i1 undef, label %while.cond.i.preheader.split, label %while.cond.outer.i.loopexit.split

while.cond.i.preheader.split:
  br label %while.cond.i

while.cond.i:
  br i1 true, label %while.cond.i, label %while.cond.outer.i.loopexit

while.cond.outer.i.loopexit:
  br label %while.cond.outer.i.loopexit.split

while.cond.outer.i.loopexit.split:
  br i1 false, label %while.cond.i.preheader, label %Func2.exit

Func2.exit:
  br label %while.body

while.body:
  br i1 false, label %while.body, label %while.end

while.end:
  br label %for.end78

for.end78:
  br i1 undef, label %Proc2.exit, label %for.cond.i.preheader

for.cond.i.preheader:
  br label %for.cond.i

for.cond.i:
  br label %for.cond.i

Proc2.exit:
  br label %for.cond31

for.end94:
  ret void
}

; Test UnloopUpdater::removeBlocksFromAncestors.
;
; Check that the loop backedge is removed from the middle loop 1699,
; but not the inner loop 1676.
; CHECK: while.body1694:
; CHECK:   br label %while.cond1676
; CHECK: while.end1699:
; CHECK:   br label %sw.default1711
define void @removeSubloopBlocks() nounwind {
entry:
  br label %tryagain.outer

tryagain.outer:                                   ; preds = %sw.bb304, %entry
  br label %tryagain

tryagain:                                         ; preds = %while.end1699, %tryagain.outer
  br i1 undef, label %sw.bb1669, label %sw.bb304

sw.bb304:                                         ; preds = %tryagain
  br i1 undef, label %return, label %tryagain.outer

sw.bb1669:                                        ; preds = %tryagain
  br i1 undef, label %sw.default1711, label %while.cond1676

while.cond1676:                                   ; preds = %while.body1694, %sw.bb1669
  br i1 undef, label %while.end1699, label %while.body1694

while.body1694:                                   ; preds = %while.cond1676
  br label %while.cond1676

while.end1699:                                    ; preds = %while.cond1676
  br i1 false, label %tryagain, label %sw.default1711

sw.default1711:                                   ; preds = %while.end1699, %sw.bb1669, %tryagain
  br label %defchar

defchar:                                          ; preds = %sw.default1711, %sw.bb376
  br i1 undef, label %if.end2413, label %if.then2368

if.then2368:                                      ; preds = %defchar
  unreachable

if.end2413:                                       ; preds = %defchar
  unreachable

return:                                           ; preds = %sw.bb304
  ret void
}

; PR11335: the most deeply nested block should be removed from the outer loop.
; CHECK: @removeSubloopBlocks2
; CHECK: for.cond3:
; CHECK-NOT: br
; CHECK: ret void
define void @removeSubloopBlocks2() nounwind {
entry:
  %tobool.i = icmp ne i32 undef, 0
  br label %lbl_616

lbl_616.loopexit:                                 ; preds = %for.cond
  br label %lbl_616

lbl_616:                                          ; preds = %lbl_616.loopexit, %entry
  br label %for.cond

for.cond:                                         ; preds = %for.cond3, %lbl_616
  br i1 false, label %for.cond1.preheader, label %lbl_616.loopexit

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond1.loopexit:                               ; preds = %for.cond.i
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1.loopexit, %for.cond1.preheader
  br i1 false, label %for.body2, label %for.cond3

for.body2:                                        ; preds = %for.cond1
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.cond.i, %for.body2
  br i1 %tobool.i, label %for.cond.i, label %for.cond1.loopexit

for.cond3:                                        ; preds = %for.cond1
  br i1 false, label %for.cond, label %if.end

if.end:                                           ; preds = %for.cond3
  ret void
}
