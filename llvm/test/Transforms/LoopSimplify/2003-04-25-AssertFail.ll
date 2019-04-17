; This testcase exposed a problem with the loop identification pass (LoopInfo).
; Basically, it was incorrectly calculating the loop nesting information.
;
; RUN: opt < %s -loop-simplify

define i32 @yylex() {
	br label %loopentry.0
loopentry.0:		; preds = %else.4, %0
	br label %loopexit.2
loopexit.2:		; preds = %else.4, %loopexit.2, %loopentry.0
	br i1 false, label %loopexit.2, label %else.4
yy_find_action:		; preds = %else.4
	br label %else.4
else.4:		; preds = %yy_find_action, %loopexit.2
	switch i32 0, label %loopexit.2 [
		 i32 2, label %yy_find_action
		 i32 0, label %loopentry.0
	]
}

