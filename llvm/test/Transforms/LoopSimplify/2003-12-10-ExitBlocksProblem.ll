; LoopSimplify is breaking LICM on this testcase because the exit blocks from
; the loop are reachable from more than just the exit nodes: the exit blocks
; have predecessors from outside of the loop!
;
; This is distilled from a monsterous crafty example.

; RUN: opt < %s -passes=licm  -disable-output


@G = weak global i32 0		; <ptr> [#uses=7]

define i32 @main() {
entry:
	store i32 123, ptr @G
	br label %loopentry.i
loopentry.i:		; preds = %endif.1.i, %entry
	%tmp.0.i = load i32, ptr @G		; <i32> [#uses=1]
	%tmp.1.i = icmp eq i32 %tmp.0.i, 123		; <i1> [#uses=1]
	br i1 %tmp.1.i, label %Out.i, label %endif.0.i
endif.0.i:		; preds = %loopentry.i
	%tmp.3.i = load i32, ptr @G		; <i32> [#uses=1]
	%tmp.4.i = icmp eq i32 %tmp.3.i, 126		; <i1> [#uses=1]
	br i1 %tmp.4.i, label %ExitBlock.i, label %endif.1.i
endif.1.i:		; preds = %endif.0.i
	%tmp.6.i = load i32, ptr @G		; <i32> [#uses=1]
	%inc.i = add i32 %tmp.6.i, 1		; <i32> [#uses=1]
	store i32 %inc.i, ptr @G
	br label %loopentry.i
Out.i:		; preds = %loopentry.i
	store i32 0, ptr @G
	br label %ExitBlock.i
ExitBlock.i:		; preds = %Out.i, %endif.0.i
	%tmp.7.i = load i32, ptr @G		; <i32> [#uses=1]
	ret i32 %tmp.7.i
}

