; This testcase was incorrectly computing that the loopentry.7 loop was
; not a child of the loopentry.6 loop.
;
; RUN: opt < %s -analyze -loops | \
; RUN:   grep "^            Loop at depth 4 containing: %loopentry.7<header><latch><exiting>"

define void @getAndMoveToFrontDecode() {
	br label %endif.2

endif.2:		; preds = %loopexit.5, %0
	br i1 false, label %loopentry.5, label %UnifiedExitNode

loopentry.5:		; preds = %loopexit.6, %endif.2
	br i1 false, label %loopentry.6, label %UnifiedExitNode

loopentry.6:		; preds = %loopentry.7, %loopentry.5
	br i1 false, label %loopentry.7, label %loopexit.6

loopentry.7:		; preds = %loopentry.7, %loopentry.6
	br i1 false, label %loopentry.7, label %loopentry.6

loopexit.6:		; preds = %loopentry.6
	br i1 false, label %loopentry.5, label %loopexit.5

loopexit.5:		; preds = %loopexit.6
	br i1 false, label %endif.2, label %UnifiedExitNode

UnifiedExitNode:		; preds = %loopexit.5, %loopentry.5, %endif.2
	ret void
}
