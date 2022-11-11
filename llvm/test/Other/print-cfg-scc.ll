; RUN: opt < %s -passes=print-cfg-sccs -disable-output 2>&1 | FileCheck %s

; CHECK: SCC #1: %UnifiedExitNode
; CHECK: SCC #2: %loopexit.5, %loopexit.6, %loopentry.7, %loopentry.6, %loopentry.5, %endif.2
; CHECK: SCC #3: %entry

define void @getAndMoveToFrontDecode() {
entry:	
	br label %endif.2

endif.2:		; preds = %loopexit.5, %entry
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
