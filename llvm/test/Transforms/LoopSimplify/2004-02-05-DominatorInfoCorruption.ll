; RUN: opt < %s -loop-simplify -verify -licm -disable-output

define void @.subst_48() {
entry:
	br label %loopentry.0
loopentry.0:		; preds = %loopentry.0, %entry
	br i1 false, label %loopentry.0, label %loopentry.2
loopentry.2:		; preds = %loopentry.2, %loopentry.0
	%tmp.968 = icmp sle i32 0, 3		; <i1> [#uses=1]
	br i1 %tmp.968, label %loopentry.2, label %UnifiedReturnBlock
UnifiedReturnBlock:		; preds = %loopentry.2
	ret void
}

