; RUN: llvm-as < %s | opt -tailduplicate -disable-output

define i32 @sell_haggle() {
entry:
	br i1 false, label %then.5, label %UnifiedExitNode
then.5:		; preds = %entry
	br i1 false, label %loopentry.1.preheader, label %else.1
else.1:		; preds = %then.5
	br label %loopentry.1.preheader
loopentry.1.preheader:		; preds = %else.1, %then.5
	%final_ask.0 = phi i32 [ 0, %else.1 ], [ 0, %then.5 ]		; <i32> [#uses=2]
	br label %loopentry.1
loopentry.1:		; preds = %endif.17, %loopentry.1.preheader
	switch i32 0, label %UnifiedExitNode [
		 i32 2, label %UnifiedExitNode
		 i32 1, label %endif.16
	]
endif.16:		; preds = %loopentry.1
	br i1 false, label %then.17, label %UnifiedExitNode
then.17:		; preds = %endif.16
	br i1 false, label %then.18, label %endif.17
then.18:		; preds = %then.17
	br i1 false, label %endif.17, label %UnifiedExitNode
endif.17:		; preds = %then.18, %then.17
	%cur_ask.3 = phi i32 [ %final_ask.0, %then.17 ], [ %final_ask.0, %then.18 ]		; <i32> [#uses=0]
	br i1 false, label %loopentry.1, label %UnifiedExitNode
UnifiedExitNode:		; preds = %endif.17, %then.18, %endif.16, %loopentry.1, %loopentry.1, %entry
	ret i32 0
}
